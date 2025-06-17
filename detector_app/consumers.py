# detector_app/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ScanStatusConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.scan_id = self.scope['url_route']['kwargs']['scan_id']
        self.group_name = f'scan_{self.scan_id}'

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def scan_status_update(self, event):
        await self.send(text_data=json.dumps({
            'status': event['status'],
            'message': event.get('message', '')
        }))
