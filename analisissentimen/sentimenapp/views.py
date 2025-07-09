from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
import pandas as pd
from django.core.paginator import Paginator
from .models import DataTeks, Stopword, Slangword, PreprocessingModel
from .forms import StopwordForm, SlangwordForm
from django.db import transaction
from django.views.decorators.http import require_POST
from django.contrib import messages
from .utils import Preprocessing, Klasifikasi, Labeling
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import joblib

logger = logging.getLogger(__name__)

def inputdata(request):
    context = {
        'title': 'Input Data'
    }


    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']

    
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File harus berformat CSV!')
            return redirect('sentimen.inputdata')

        
        if csv_file.size > 5 * 1024 * 1024:
            messages.error(request, 'Ukuran file terlalu besar! Maksimal 5MB.')
            return redirect('sentimen.inputdata')

        try:
          
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_file, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_file, encoding='cp1252')

            df.columns = df.columns.str.strip()

            if 'teks' not in df.columns:
                messages.error(request, f"Kolom 'teks' tidak ditemukan. Kolom tersedia: {', '.join(df.columns)}")
                return redirect('sentimen.inputdata')

            if df.empty:
                messages.error(request, 'File CSV kosong atau tidak memiliki data.')
                return redirect('sentimen.inputdata')

        
            df['teks'] = df['teks'].astype(str).str.strip()
            df = df[df['teks'].notna() & (df['teks'] != '') & (df['teks'] != 'nan')]

            if df.empty:
                messages.error(request, 'Tidak ada data teks yang valid.')
                return redirect('sentimen.inputdata')

            initial_count = len(df)
            df = df.drop_duplicates(subset=['teks'])
            duplicate_count = initial_count - len(df)

            existing_texts = set(DataTeks.objects.values_list('teks', flat=True))
            new_data = df[~df['teks'].isin(existing_texts)]
            existing_data_count = len(df) - len(new_data)

            if not new_data.empty:
                with transaction.atomic():
                    DataTeks.objects.bulk_create(
                        [DataTeks(teks=row['teks']) for _, row in new_data.iterrows()],
                        batch_size=1000
                    )
                msg = f"{len(new_data)} data baru disimpan!"
                if duplicate_count > 0:
                    msg += f" ({duplicate_count} duplikat dihapus)"
                if existing_data_count > 0:
                    msg += f" ({existing_data_count} data sudah ada sebelumnya)"
                messages.success(request, msg)
            else:
                messages.warning(request, 'Semua data sudah ada di database.')

            return redirect('sentimen.inputdata')

        except Exception as e:
            logger.error(f'CSV Error: {e}')
            messages.error(request, f'Terjadi kesalahan saat membaca file: {str(e)}')
            return redirect('sentimen.inputdata')
    elif request.method == "POST" and "drop" in request.POST:
            truncate_table(DataTeks)
            messages.success(request, "Data berhasil dihapus!")
            return redirect('sentimen.inputdata')

    all_data = DataTeks.objects.all().order_by('id')
    paginator = Paginator(all_data, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context.update({
        'csv_columns': ['teks'],
        'csv_rows': page_obj,
        'data_count': paginator.count,
        'page_obj': page_obj,
    })

    return render(request, 'inputdata/inputdata.html', context)

# Stopword
def stopword(request):
    context = {
        'title': 'Stopword'
    }
    if request.method == "POST" and 'stopword' in request.FILES:
        csv_file = request.FILES['stopword']
        if not csv_file:
            messages.error(request, 'File harus berformat CSV! atau Silakan Upload terlebih dahulu')
            return redirect('sentimen.stopword')
        if csv_file.size > 5 * 1024 * 1024:
            messages.error(request, 'Ukuran file terlalu besar! Maksimal 5MB.')
            return redirect('sentimen.stopword')
        try:
            data = pd.read_csv(csv_file)
            if 'stopword' not in data.columns:
                messages.error(request, "Kolom 'stopword' harus ada di file CSV!")
                return redirect('sentimen.stopword')
            stopwords = list(data['stopword'])
            for x in stopwords :
                with transaction.atomic():
                    Stopword.objects.filter(stopwords=x).delete()
                    Stopword.objects.create(stopwords=x)
            messages.success(request, "Stopword berhasil ditambahkan!")
            return redirect('sentimen.stopword')

        except Exception as e:
            messages.error(request, f'Terjadi kesalahan saat membaca file: {str(e)}')
            return redirect('sentimen.stopword')

    stopwords = Stopword.objects.all().order_by('id')
    paginator = Paginator(stopwords, 10)

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'stopword/index.html', context | {'stopwords': stopwords, 'page_obj': page_obj})


@require_POST
def stopword_create(request):
    form = StopwordForm(request.POST)
    if form.is_valid():
        form.save()
        messages.success(request, "Stopword berhasil ditambahkan!")
    return redirect('sentimen.stopword')

@require_POST
def stopword_edit(request, id):
    stopword = get_object_or_404(Stopword, id=id)
    form = StopwordForm(request.POST, instance=stopword)
    if form.is_valid():
        form.save()
        messages.success(request, "Stopword berhasil diubah!")
    return redirect('sentimen.stopword')

@require_POST
def stopword_delete(request, id):
    stopword = get_object_or_404(Stopword, id=id)
    stopword.delete()
    messages.success(request, "Stopword berhasil dihapus!")
    return redirect('sentimen.stopword')

# Slangword
def slangword(request):
    if request.method == "POST" and 'slangword' in request.FILES:
        csv_file = request.FILES['slangword']
        if not csv_file:
            messages.error(request, 'File harus berformat CSV! atau Silakan Upload terlebih dahulu')
            return redirect('sentimen.slangword')
        if csv_file.size > 5 * 1024 * 1024:
            messages.error(request, 'Ukuran file terlalu besar! Maksimal 5MB.')
            return redirect('sentimen.slangword')
        try:
            data = pd.read_csv(csv_file)
            print(data.columns)
            if 'kata_baku' not in data.columns or 'kata_tidak_baku' not in data.columns:
                messages.error(request, "Kolom 'kata_baku' dan 'kata_tidak_baku' harus ada di file CSV!")
                return redirect('sentimen.slangword')
            
            
            for katabaku, katatidakbaku in zip(data['kata_baku'], data['kata_tidak_baku']):
                with transaction.atomic():
                    Slangword.objects.filter(katabaku=katabaku).delete()
                    Slangword.objects.filter(katatidakbaku=katatidakbaku).delete()
                    Slangword.objects.create(katabaku=katabaku, katatidakbaku=katatidakbaku)
            messages.success(request, "Slangword berhasil ditambahkan!")
            return redirect('sentimen.slangword')

        except Exception as e:
            messages.error(request, f'Terjadi kesalahan saat membaca file: {str(e)}')
            return redirect('sentimen.slangword')
    if request.method == "POST":
        katabaku = request.POST.get("katabaku")
        katatidakbaku = request.POST.get("katatidakbaku")
        
        if katabaku and katatidakbaku:
            Slangword.objects.create(katabaku=katabaku, katatidakbaku=katatidakbaku)
            return redirect('sentimen.slangword')
        
    slangwords = Slangword.objects.all().order_by('id')
    paginator = Paginator(slangwords, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'title': 'Slangword',
        'slangwords': slangwords,
        'page_obj': page_obj
    }
    return render(request, 'slangword/index.html', context)

@require_POST
def slangword_create(request):
    form = SlangwordForm(request.POST)
    if form.is_valid():
        form.save()
    return redirect('sentimen.slangword')

@require_POST
def slangword_edit(request, id):
    slangword = get_object_or_404(Slangword, id=id)
    form = SlangwordForm(request.POST, instance=slangword)
    if form.is_valid():
        form.save()
    return redirect('sentimen.slangword')

@require_POST
def slangword_delete(request, id):
    slangword = get_object_or_404(Slangword, id=id)
    slangword.delete()
    return redirect('sentimen.slangword')
def preprocessing(request):
    title = "Preprocessing"
    data = PreprocessingModel.objects.all().order_by('id')
    paginator = Paginator(data, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    message = None

    if request.method == "POST" and "proses" in request.POST:
        data_teks_list = list(DataTeks.objects.all().order_by('id'))

        if not data_teks_list:
            message = "Tidak ada data teks yang ditemukan untuk diproses."
            return render(request, 'preprocessing/preprocessing.html', {
                'page_obj': page_obj,
                'message': message,
                'title': title
            })

        pre = Preprocessing()
        bulk_preprocessing_create = []

        texts_to_process = [obj.teks for obj in data_teks_list]
        cleaned_texts_corpus = pre.process_corpus(texts_to_process)

        for i, data_teks_obj in enumerate(data_teks_list):
            teks_asli = data_teks_obj.teks
            cleaned_text = cleaned_texts_corpus[i]

            case_fold = pre._case_folding(teks_asli)
            no_symbol = pre._remove_symbols(case_fold)
            tokens = pre._tokenize(no_symbol)
            slang_fixed = pre._replace_slang(tokens)
            stop_removed = pre._remove_stopwords(slang_fixed)
            stemmed_tokens = pre._stemming(stop_removed)

        
            print("Stopword aktif:", pre.stopwords)
            print("Tokens sebelum stopword:", slang_fixed)
            print("Tokens sesudah stopword:", stop_removed)

            bulk_preprocessing_create.append(
                PreprocessingModel(
                    teks_awal=data_teks_obj,
                    case_folding=case_fold,
                    symbol_removal=no_symbol,
                    tokenizing=' '.join(tokens),
                    slangword_removal=' '.join(slang_fixed),
                    stopword_removal=' '.join(stop_removed),
                    stemming=' '.join(stemmed_tokens),
                    text_bersih=cleaned_text
                )
            )

        with transaction.atomic():
            PreprocessingModel.objects.bulk_create(bulk_preprocessing_create, ignore_conflicts=True)
            PreprocessingModel.objects.filter(text_bersih="").delete()

        message = 'Preprocessing dan pelabelan otomatis selesai untuk seluruh data.'
        data = PreprocessingModel.objects.all().order_by('id')
        paginator = Paginator(data, 10)
        page_obj = paginator.get_page(1)

    return render(request, 'preprocessing/preprocessing.html', {
        'page_obj': page_obj,
        'title': title,
        'message': message,
    })
def labelisasi(request):
    title = 'Labelisasi'
    message = None
    page_obj = None

    if request.method == "POST":
        preprocessing_data_for_processing = PreprocessingModel.objects.all().order_by('id')

        if not preprocessing_data_for_processing.exists():
            message = "Tidak ada data preprocessing yang ditemukan untuk dilabeli."
        else:
            lab = Labeling()
            bulk_label_updates = []

            try:
                for pp_obj in preprocessing_data_for_processing:
                    if not pp_obj.text_bersih or not pp_obj.text_bersih.strip():
                        label = 'netral'
                        print(f"Warning: Cleaned text for PreprocessingModel ID {pp_obj.id} is empty, assigning 'netral'.")
                    else:
                        label = lab.label_by_textblob_fast(pp_obj.text_bersih)

                    
                    pp_obj.label = label
                    bulk_label_updates.append(pp_obj)

                if not bulk_label_updates:
                    message = "Tidak ada label yang dihasilkan untuk diperbarui."
                else:
                    with transaction.atomic():
                        PreprocessingModel.objects.bulk_update(bulk_label_updates, ['label'])

                    message = 'Pelabelan otomatis selesai untuk seluruh data preprocessing.'

            except Exception as e:
                print(f"Error during bulk labeling: {e}")
                message = 'Gagal melakukan pelabelan otomatis. Periksa konfigurasi model atau batasan layanan labeling.'
    
    
    data_to_display = PreprocessingModel.objects.select_related('teks_awal').all().order_by('id')
    paginator = Paginator(data_to_display, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'preprocessing/labelisasi.html', {
        'title': title,
        'message': message,
        'page_obj': page_obj,
    })

def tfidf(request):
    data = PreprocessingModel.objects.all()
    texts = [item.text_bersih for item in data]
    
   
    if not texts:
        return render(request, 'hasil/tfidf.html', {
            'error': 'Data teks bersih untuk TF-IDF belum tersedia. Jalankan preprocessing terlebih dahulu.',
            'title': 'Hasil TF-IDF'
        })

    model = Klasifikasi()
    try:
        X = model.vectorizer.transform(texts) 
    except AttributeError:
        return render(request, 'hasil/tfidf.html', {
            'error': 'Vectorizer belum dilatih. Harap jalankan proses Klasifikasi terlebih dahulu.',
            'title': 'Hasil TF-IDF'
        })
        
    feature_names = model.vectorizer.get_feature_names_out()
    tfidf_matrix = X.toarray() 

    tfidf_display = []
 
    for i, row in enumerate(tfidf_matrix[:10]): 
        doc_features = []
        sorted_features = sorted([(feature_names[j], score) for j, score in enumerate(row) if score > 0], key=lambda x: x[1], reverse=True)
        
        for term, score in sorted_features:
            doc_features.append({
                'term': term,
                'score': round(score, 4)
            })
        tfidf_display.append({
            'doc_index': i + 1,
            'features': doc_features
        })
    
    return render(request, 'hasil/tfidf.html', {
        'tfidf_display': tfidf_display,
        'title': 'Hasil TF-IDF'
    })
def klasifikasi(request):
    title = 'Klasifikasi'

    
    data = PreprocessingModel.objects.exclude(label__isnull=True).exclude(text_bersih__isnull=True)


    texts = [item.text_bersih for item in data]
    labels = [item.label for item in data]

    if not texts or not labels:
        return render(request, 'klasifikasi/klasifikasi.html', {
            'error': 'Data teks bersih atau label untuk klasifikasi belum tersedia. Jalankan preprocessing dan labelisasi terlebih dahulu.',
            'title': title
        })

    
    classifier_model = Klasifikasi()

    try:
      
        cm, report, acc = classifier_model.train_model(texts, labels)
    except Exception as e:
        print(f"Error during model training: {e}")
        return render(request, 'klasifikasi/klasifikasi.html', {
            'error': f'Gagal melatih model: {e}. Pastikan data cukup dan format label benar.',
            'title': title
        })

   
    cm_percent = np.round((cm / cm.sum(axis=1, keepdims=True)) * 100, 2) if cm.sum() > 0 else np.array([])

   
    labels_class = list(classifier_model.model.classes_)

   
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for key in ['precision', 'recall', 'f1-score']:
                if key in metrics:
                    metrics[key] = round(metrics[key] * 100, 2)

    accuracy_percent = round(acc * 100, 2)

   
    request.session['labels_class'] = labels_class
    request.session['cm_percent'] = cm_percent.tolist()

    return render(request, 'klasifikasi/klasifikasi.html', {
        'accuracy': accuracy_percent,
        'report': report,
        'confusion_matrix': cm_percent,
        'class_labels': labels_class,
        'title': title
    })


def evaluasi(request):
    confusion_matrix = request.session.get('cm_percent')
    labels_class = request.session.get('labels_class')
    if not confusion_matrix or not labels_class:
        return render(request, 'hasil/evaluasi.html', {
            'error': 'Data untuk evaluasi belum tersedia atau belum memproses klasifikasi.'
        })
    
    return render(request, 'hasil/evaluasi.html',
                  {
                      'confusion_matrix': confusion_matrix,
                      'labels_class': labels_class,
                      'title' : 'Evaluasi'
                  })

def truncate_table(model):
    model.objects.all().delete()
    
def kalimat(request):
    title = "Test Kalimat"
    prediction_result = None # Initialize to None

    if request.method == "POST":
        input_kalimat = request.POST.get('kalimat')
        
        if not input_kalimat or not input_kalimat.strip():
            
            prediction_result = {
                'kalimat': input_kalimat,
                'sentimen': 'Tidak dapat memproses kalimat kosong.'
            }
        else:
            try:
                model = Klasifikasi()
                pre = Preprocessing()
                
              
                processed_kalimat = pre.clean_text(input_kalimat)
                
                if not processed_kalimat.strip():
                    prediction_result = {
                        'kalimat': input_kalimat,
                        'sentimen': 'Kalimat setelah preprocessing kosong atau hanya noise.'
                    }
                else:
                    prediction = model.predict(processed_kalimat)
                    
                    
                    prediction_result = {
                        'kalimat': input_kalimat,
                        'sentimen': prediction
                    }
            except FileNotFoundError:
                prediction_result = {
                    'kalimat': input_kalimat,
                    'sentimen': 'Model belum terlatih atau tidak ditemukan. Harap jalankan klasifikasi terlebih dahulu.'
                }
            except Exception as e:
            
                prediction_result = {
                    'kalimat': input_kalimat,
                    'sentimen': f'Terjadi kesalahan saat prediksi: {e}'
                }

    return render(request, 'kalimat/kalimat.html', {
        'title': title,
        'prediction': prediction_result 
    })