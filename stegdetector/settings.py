import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-5a+9lf9l98+&(@%(yubq)cgu!5!0)2vtm&%e7z5=401j$%uje!'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'accounts',
    'pdf_parser',
    'pdf_stego',
    'detector_app',
    'ai', 
    'django_celery_results',
    'rest_framework',
    'channels',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'stegdetector.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [ os.path.join(BASE_DIR, 'templates') ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'stegdetector.wsgi.application'

ASGI_APPLICATION = 'stegdetector.asgi.application'

# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'stegodetector_db',  
        'USER': 'stegodetector_user',
        'PASSWORD': 'steg0_passw0rd',
        'HOST': 'localhost',
        'PORT': '5432',
        'OPTIONS': {
            'options': '-c search_path=stegodetector_schema'
        }
    }
}






# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Media files (uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Ensure the log directory exists
LOG_DIR = BASE_DIR / 'logs' / 'log'
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    
    'formatters': {
        'verbose': {
            'format': '{asctime} [{levelname}] {name} — {message}',
            'style': '{',
        },
        'simple': {
            'format': '[{levelname}] {message}',
            'style': '{',
        },
    },

    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOG_DIR / 'django.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        }
    },

    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },

    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.server': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'ai': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        # Custom apps can have their own logger
        'pdf_stego': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'pdf_detector': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}



# Celery Configuration
CELERY_BROKER_URL = 'redis://localhost:6379/0' # Or your RabbitMQ URL
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'Africa/Dar_es_Salaam' # Your timezone
CELERY_ENABLE_UTC = True

# ML Model Settings
MODELS_DIR = os.path.join(BASE_DIR, 'ai', 'models')
ML_MODEL_DIR = os.path.join(BASE_DIR, 'detector_app', 'ml', 'models')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
UPLOAD_DIR = os.path.join(MEDIA_ROOT, 'uploads')
PDF_STEGANO_MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'stego')
PDF_STEGANO_CACHE_DIR = os.path.join(BASE_DIR, 'ml_models', 'cache')

DATASETS_BASE_DIR = os.path.join(BASE_DIR, 'datasets')

# Ensure directories exist
os.makedirs(ML_MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASETS_BASE_DIR, exist_ok=True)



# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB


# Cache (for model caching)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'pdf-detector-cache',
    }
}


CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [("127.0.0.1", 6379)],
        },
    },
}


LOGIN_REDIRECT_URL = 'ai:ml_dashboard'
LOGOUT_REDIRECT_URL = 'accounts:login'