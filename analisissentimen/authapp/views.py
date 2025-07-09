from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from sentimenapp.models import DataTeks
from sentimenapp.models import Stopword
from sentimenapp.models import Slangword


def login_view(request):
    if request.user.is_authenticated:
        return redirect('sentimen.index')

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        
        user = authenticate(request, username=username, password=password)
        if user is not None :
            login(request, user)
            return redirect('sentimen.index')
        else:
            messages.error(request, 'Username atau password salah, atau Anda tidak punya akses.')

    return render(request, 'authapp/login.html')

def logout_view(request):
    logout(request)
    return redirect('auth.login')

@login_required(login_url='auth.login')
def dashboard_view(request):
    context = {
        'user': request.user,
        'total_data': DataTeks.objects.count(),
        'total_stopword': Stopword.objects.count(),
        'total_slangword': Slangword.objects.count(),
        'title': 'Dashboard'

    }
    return render(request, 'dashboard/index.html', context)