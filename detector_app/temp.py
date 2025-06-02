    '''
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('scan/<uuid:scan_id>/', views.scan_detail, name='scan_detail'),
    path('history/', views.scan_history, name='scan_history'),
    path('delete/<uuid:scan_id>/', views.delete_scan, name='delete_scan'),
    ''',