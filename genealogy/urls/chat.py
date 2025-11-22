"""URL configuration for chat interface"""

from django.urls import path

from ..views import chat

urlpatterns = [
    path('', chat.chat_index, name='chat_index'),
    path('new/', chat.new_conversation, name='chat_new'),
    path('<uuid:conversation_id>/', chat.conversation_detail, name='chat_conversation'),
    path('<uuid:conversation_id>/stream/', chat.stream_message, name='chat_stream'),
]
