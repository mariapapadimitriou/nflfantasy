"""
URL configuration for predictions app
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/load-data/', views.load_data, name='load_data'),
    path('api/train-model/', views.train_model_view, name='train_model'),
    path('api/retrain-model/', views.retrain_model_view, name='retrain_model'),
    path('api/predict-week/', views.predict_week_view, name='predict_week'),
    path('api/export-predictions/', views.export_predictions, name='export_predictions'),
    path('api/check-model/', views.check_model_exists, name='check_model'),
    path('api/feature-data/', views.get_feature_data, name='feature_data'),
]

