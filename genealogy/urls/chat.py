"""URL configuration for chat interface"""

from django.urls import path

from ..views import chat

urlpatterns = [
    path('', chat.chat_index, name='chat_index'),
    path('new/', chat.new_conversation, name='chat_new'),
    path('<uuid:conversation_id>/', chat.conversation_detail, name='chat_conversation'),
    path('<uuid:conversation_id>/send/', chat.send_message, name='chat_send'),
    path('<uuid:conversation_id>/set-template/', chat.set_prompt_template, name='chat_set_template'),
    path('stream/<uuid:message_id>/', chat.stream_events, name='chat_stream'),
]
