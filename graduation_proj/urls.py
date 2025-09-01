from django.conf.urls.static import static
from django.conf import settings

from django.urls import path, include


urlpatterns = [
    path('', include('gaeyeon.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


