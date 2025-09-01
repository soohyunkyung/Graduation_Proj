from . import views
from django.urls import path


urlpatterns = [
    path('upload/', views.upload_view),
    path('survey/', views.survey_view)
]

