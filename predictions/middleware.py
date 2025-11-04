"""
Custom middleware to handle CSRF errors and return JSON responses
"""
from django.http import JsonResponse
from django.middleware.csrf import CsrfViewMiddleware
from django.utils.deprecation import MiddlewareMixin


class JsonCsrfViewMiddleware(CsrfViewMiddleware):
    """
    Custom CSRF middleware that returns JSON errors instead of HTML
    for API endpoints
    """
    
    def process_view(self, request, view_func, view_args, view_kwargs):
        # Check if this is an API endpoint
        if request.path.startswith('/api/'):
            try:
                return super().process_view(request, view_func, view_args, view_kwargs)
            except Exception as e:
                # Return JSON error instead of HTML
                return JsonResponse({
                    'success': False,
                    'message': 'CSRF verification failed. Please refresh the page and try again.',
                    'error': 'CSRF_ERROR'
                }, status=403)


class JsonErrorMiddleware(MiddlewareMixin):
    """
    Middleware to catch exceptions and return JSON responses for API endpoints
    """
    
    def process_exception(self, request, exception):
        # Only handle API endpoints
        if request.path.startswith('/api/'):
            import traceback
            return JsonResponse({
                'success': False,
                'message': f'Server error: {str(exception)}',
                'error': type(exception).__name__,
                'traceback': traceback.format_exc() if request.GET.get('debug') else None
            }, status=500)
        # Let other exceptions propagate normally
        return None

