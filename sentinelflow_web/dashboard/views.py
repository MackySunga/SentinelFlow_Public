from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render

from .services import analyze_uploaded_file, available_reports, load_latest_analysis, load_notebook_metrics, model_registry_status


def home(request):
    return render(request, 'dashboard/home.html', {
        'model_status': model_registry_status(),
    })


def upload_dataset(request):
    context = {}
    if request.method == 'POST':
        uploaded = request.FILES.get('dataset')
        if not uploaded:
            messages.error(request, 'Please select a CSV, TSV, TXT, or Parquet file.')
            return redirect('upload_dataset')
        allowed = ['.csv', '.tsv', '.txt', '.parquet', '.pq']
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in allowed:
            messages.error(request, f'Unsupported file type: {suffix}. Use CSV, TSV, TXT, or Parquet.')
            return redirect('upload_dataset')
        upload_dir = Path(settings.MEDIA_ROOT) / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded.name
        with open(save_path, 'wb+') as destination:
            for chunk in uploaded.chunks():
                destination.write(chunk)
        result = analyze_uploaded_file(save_path)
        if result.ok:
            messages.success(request, result.message)
            context['analysis'] = result.context
        else:
            messages.error(request, result.message)
            context['analysis_error'] = result.message
    else:
        context['analysis'] = load_latest_analysis()
    return render(request, 'dashboard/upload.html', context)


def dashboard(request):
    return render(request, 'dashboard/dashboard.html', {
        'analysis': load_latest_analysis(),
        'metrics': load_notebook_metrics(),
        'model_status': model_registry_status(),
    })


def model_status(request):
    return render(request, 'dashboard/models.html', {
        'model_status': model_registry_status(),
        'metrics': load_notebook_metrics(),
    })


def reports(request):
    return render(request, 'dashboard/reports.html', {
        'reports': available_reports(),
        'metrics': load_notebook_metrics(),
    })
