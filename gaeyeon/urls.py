from . import views
from django.urls import path


urlpatterns = [
    path('', views.upload_view,name='upload'),
    path('survey/', views.survey_view,name='survey')
]

