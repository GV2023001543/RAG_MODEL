"""TeamDino: a document-first study workspace."""
from __future__ import annotations
import html
import hmac
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_javascript import st_javascript

# Streamlit reruns this entrypoint without necessarily re-importing the backend.
# Reload local configuration here so newly added .env values become visible.
load_dotenv(Path(__file__).with_name('.env'), override=False)

from langgraph_rag_backend import (
    delete_conversation, export_conversation_txt, generate_chat_title, get_chatbot,
    get_thread_title, ingest_file,
    remove_thread_document, retrieve_user_threads, set_thread_title, storage,
    thread_document_metadata,
)
from rag_providers import (
    chat_model_status, content_text, friendly_error,
    save_user_provider_key, list_user_provider_keys,
    delete_user_provider_key, set_user_provider_key_enabled,
    save_system_provider_key, list_system_provider_keys, delete_system_provider_key,
    set_system_provider_key_enabled, revalidate_provider_key,
    list_provider_configs, save_provider_config,
    save_search_key, search_key_status, delete_search_key,
)
from browser_history import USER_KEY, build_snapshot, history_key, restore_snapshot, valid_user_id

BRAND = 'TeamDino'
SUPPORTED_TYPES = ['pdf', 'docx', 'pptx', 'xlsx', 'xls', 'csv', 'txt', 'md']
logger = logging.getLogger(__name__)
st.set_page_config(page_title='TeamDino | Your study workspace', page_icon=':material/auto_stories:',
                   layout='wide', initial_sidebar_state='expanded')


def _server_secret(name: str) -> str:
    """Read a local environment variable or root-level Streamlit Cloud secret."""
    value = os.getenv(name, '').strip()
    if value:
        return value
    try:
        value = st.secrets.get(name, '')
    except FileNotFoundError:
        return ''
    return str(value).strip() if value is not None else ''


def _css():
    css = Path(__file__).with_name('assets').joinpath('workspace.css').read_text(encoding='utf-8')
    if st.session_state.get('appearance') == 'dark':
        css += '''
        :root { --bg:#15211c; --side:#101b16; --panel:#1c2c24; --soft:#213329;
          --ink:#e5eee3; --muted:#a0b49f; --line:#324839; --green:#80be8e; --accent:#294633;
          --glow-a:#20372b; --glow-b:#182a21; }
        .brand-mark, .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { color:#14251a; }
        [data-testid="stChatInput"] textarea { -webkit-text-fill-color:var(--ink); }
        [data-testid="stWidgetLabel"], [data-testid="stFileUploaderDropzone"] { color:var(--ink); }
        [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background:var(--panel) !important; }
        '''
    return '<style>' + css + '</style>'


def reset_chat():
    tid = f"{st.session_state['user_id']}::{uuid.uuid4()}"
    storage.ensure_thread(tid, st.session_state['user_id'])
    st.session_state['thread_id'] = tid
    st.session_state.pop('failed_request', None)
    st.session_state['uploader_version'] = st.session_state.get('uploader_version', 0) + 1
    st.query_params['chat'] = tid.split('::')[1]


def _init_user_session(browser_user_id: str = ''):
    if 'user_id' not in st.session_state:
        user_id = valid_user_id(browser_user_id) or valid_user_id(st.query_params.get('uid', '')) or str(uuid.uuid4())
        st.session_state['user_id'] = user_id
        st.query_params['uid'] = user_id
    st.session_state.setdefault('appearance', 'light')
    st.session_state.setdefault('uploader_version', 0)
    if 'thread_id' not in st.session_state:
        threads = retrieve_user_threads(st.session_state['user_id'])
        requested = f"{st.session_state['user_id']}::{st.query_params.get('chat', '')}"
        if requested in threads:
            st.session_state['thread_id'] = requested
        elif threads:
            st.session_state['thread_id'] = threads[-1]
        else:
            reset_chat()


def _browser_history_state():
    return st_javascript(f'''(() => ({{
      ready: true,
      user_id: localStorage.getItem({json.dumps(USER_KEY)}),
      history: localStorage.getItem({json.dumps('teamdino.history.current')})
    }}))()''', key='teamdino_history_read')


def _write_browser_history(user_id: str, snapshot: str) -> None:
    values = json.dumps({
        USER_KEY: user_id,
        'teamdino.history.current': snapshot,
        history_key(user_id): snapshot,
    }, ensure_ascii=False)
    st_javascript(f'''(() => {{
      const values = {values};
      for (const [key, value] of Object.entries(values)) localStorage.setItem(key, value);
      return true;
    }})()''', key='teamdino_history_write')


def _sync_browser_history(browser_state) -> None:
    if not isinstance(browser_state, dict) or not browser_state.get('ready'):
        return
    user_id = st.session_state['user_id']
    if not st.session_state.get('browser_history_loaded'):
        current = st.session_state['thread_id']
        restored = restore_snapshot(storage, user_id, browser_state.get('history'))
        st.session_state['browser_history_loaded'] = True
        if restored and not storage.messages(current) and not storage.documents(current):
            storage.delete_thread(current)
            st.session_state['thread_id'] = restored[-1]
            st.query_params['chat'] = restored[-1].split('::')[1]
        if restored:
            st.session_state['browser_history_restored'] = len(restored)
    snapshot = build_snapshot(storage, user_id)
    digest = hashlib.sha256(snapshot.encode()).hexdigest()
    if st.session_state.get('browser_history_digest') != digest:
        _write_browser_history(user_id, snapshot)
        st.session_state['browser_history_digest'] = digest


def _switch_thread(tid):
    st.session_state['thread_id'] = tid
    st.session_state.pop('failed_request', None)
    st.session_state['uploader_version'] += 1
    st.query_params['chat'] = tid.split('::')[1]


def _coerce_submission(submission):
    if submission is None:
        return '', []
    if isinstance(submission, str):
        return submission.strip(), []
    return (submission.text or '').strip(), list(submission.files or [])


def _seed_prompt(prompt):
    st.session_state['rag_chat_input'] = prompt


def _key_status_label(key):
    label = key['status'].replace('_', ' ').title()
    return f"{key['provider'].title()} · {key['masked_key']} · {label}"


def _render_key_controls(keys, owner):
    if not keys:
        st.caption('No keys saved yet.')
        return
    for key in keys:
        with st.container(border=True):
            st.caption(key.get('label') or _key_status_label(key))
            st.code(_key_status_label(key), language=None)
            columns = st.columns(3)
            toggle = set_user_provider_key_enabled if owner == 'user' else set_system_provider_key_enabled
            remove = delete_user_provider_key if owner == 'user' else delete_system_provider_key
            owner_args = (st.session_state['user_id'], key['key_id']) if owner == 'user' else (key['key_id'],)
            with columns[0]:
                if st.button('Disable' if key['enabled'] else 'Enable', key=f"toggle_{owner}_{key['key_id']}"):
                    toggle(*owner_args, not key['enabled'])
                    st.rerun()
            with columns[1]:
                if st.button('Recheck', key=f"recheck_{owner}_{key['key_id']}", help='Clear cooldown or invalid status so this key can be tried again.'):
                    revalidate_provider_key(key['key_id'])
                    st.rerun()
            with columns[2]:
                if st.button('Remove', key=f"remove_key_{owner}_{key['key_id']}"):
                    remove(*owner_args)
                    st.rerun()


@st.dialog('Workspace settings', width='large')
def settings_dialog():
    personal_tab, appearance_tab, admin_tab = st.tabs(['Personal AI keys', 'Appearance & data', 'Admin pool'])
    with personal_tab:
        st.caption('Your keys are encrypted on the server and tried before the shared system pool.')
        providers = list_provider_configs()
        enabled_names = [provider['name'] for provider in providers if provider['enabled']]
        with st.form('personal_provider_key'):
            provider = st.selectbox('Provider', enabled_names, format_func=lambda value: value.replace('_', ' ').title())
            label = st.text_input('Label (optional)', placeholder='My Groq key')
            api_key = st.text_input('API key', type='password', autocomplete='off')
            if st.form_submit_button('Save encrypted key', type='primary', use_container_width=True):
                try:
                    save_user_provider_key(st.session_state['user_id'], provider, api_key, label)
                    st.success('Key saved. Its full value will not be displayed again.')
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        st.markdown('**Saved personal keys**')
        _render_key_controls(list_user_provider_keys(st.session_state['user_id']), 'user')
        st.divider()
        tavily = search_key_status(st.session_state['user_id'])
        with st.form('personal_search_key'):
            st.caption('Optional web search')
            search_key = st.text_input('Tavily API key', type='password', autocomplete='off',
                                       placeholder=tavily['masked_key'] or 'tvly-...')
            if st.form_submit_button('Save encrypted search key', use_container_width=True):
                try:
                    save_search_key(st.session_state['user_id'], search_key)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        if tavily['configured'] and st.button('Remove Tavily key'):
            delete_search_key(st.session_state['user_id'])
            st.rerun()

    with appearance_tab:
        st.radio('Appearance', ['light', 'dark'], horizontal=True, key='appearance', format_func=str.title)
        st.divider()
        confirm = st.checkbox('Delete all conversations and their documents')
        if st.button('Delete all conversations', disabled=not confirm, use_container_width=True):
            for tid in retrieve_user_threads(st.session_state['user_id']):
                delete_conversation(tid, st.session_state['user_id'])
            reset_chat()
            st.rerun()

    with admin_tab:
        admin_password = _server_secret('ADMIN_PASSWORD')
        authenticated = st.session_state.get('admin_authenticated', False)
        if not admin_password:
            st.info('Admin access is not configured for this deployment. Open the app from your Streamlit Cloud dashboard, choose **Settings → Secrets**, add `ADMIN_PASSWORD = "your-password"`, save, and restart the app. The password field will then appear here.')
        elif not authenticated:
            locked_until = st.session_state.get('admin_locked_until', 0.0)
            locked = locked_until > time.time()
            entered = st.text_input('Admin password', type='password', autocomplete='off', disabled=locked)
            if locked:
                st.error('Too many failed attempts. Try again in a few minutes.')
            if st.button('Unlock admin pool', type='primary', disabled=locked):
                if hmac.compare_digest(entered, admin_password):
                    st.session_state['admin_authenticated'] = True
                    st.session_state['admin_failed_attempts'] = 0
                    authenticated = True
                else:
                    failures = st.session_state.get('admin_failed_attempts', 0) + 1
                    st.session_state['admin_failed_attempts'] = failures
                    if failures >= 5:
                        st.session_state['admin_locked_until'] = time.time() + 300
                    st.error('Incorrect admin password.')
        if authenticated:
            configs = list_provider_configs()
            names = [config['name'] for config in configs]
            selected = st.selectbox('Configure provider', [*names, 'Add compatible provider'])
            current = next((config for config in configs if config['name'] == selected), None)
            with st.form('provider_configuration'):
                custom_name = st.text_input('Provider name', placeholder='my_provider', disabled=current is not None,
                                            value=current['name'] if current else '')
                adapter_options = ['openai', 'groq', 'gemini', 'anthropic', 'openai_compatible']
                adapter = st.selectbox('Adapter', adapter_options,
                                       index=adapter_options.index(current['adapter']) if current else 4)
                endpoint = st.text_input('API endpoint', value=current['endpoint'] if current else '',
                                         help='Required for OpenAI-compatible providers.')
                model = st.text_input('Model name', value=current['model'] if current else '')
                col1, col2, col3 = st.columns(3)
                max_tokens = col1.number_input('Max tokens', 1, 131072, current['max_tokens'] if current else 2048)
                timeout = col2.number_input('Timeout (seconds)', 1, 300, int(current['timeout']) if current else 45)
                priority = col3.number_input('Priority', -1000, 1000, current['priority'] if current else 100)
                enabled = st.checkbox('Provider enabled', value=current['enabled'] if current else True)
                if st.form_submit_button('Save provider', type='primary', use_container_width=True):
                    try:
                        save_provider_config(custom_name, adapter, endpoint, model, max_tokens, timeout, priority, enabled)
                        st.success('Provider configuration saved.')
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            with st.form('system_provider_key'):
                system_provider = st.selectbox('System key provider', names)
                system_label = st.text_input('System key label', placeholder='Groq pool key 2')
                system_key = st.text_input('System API key', type='password', autocomplete='off')
                if st.form_submit_button('Add encrypted system key', type='primary', use_container_width=True):
                    try:
                        save_system_provider_key(system_provider, system_key, system_label)
                        st.success('System key added to the rotation pool.')
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            st.markdown('**Encrypted system keys**')
            _render_key_controls(list_system_provider_keys(), 'system')
            if st.button('Lock admin pool'):
                st.session_state['admin_authenticated'] = False
                st.rerun()


def sidebar():
    with st.sidebar:
        st.markdown('''<div class="brand"><div class="brand-mark">td.</div>
          <div><strong>TeamDino</strong><small>KNOWLEDGE WORKSPACE</small></div></div>''', unsafe_allow_html=True)
        st.button('New conversation', icon=':material/add:', type='primary', key='new_chat',
                  use_container_width=True, on_click=reset_chat)
        st.markdown('''<div class="section-label">A calmer way to study</div>
          <div class="side-note">Bring your notes, ask better questions,<br>and make room for understanding.</div>''', unsafe_allow_html=True)
        st.divider()
        if st.button('Settings', icon=':material/tune:', use_container_width=True, key='sidebar_settings'):
            settings_dialog()
        st.caption('Chat history is backed up in this browser.')


def history_panel():
    with st.container(key='history_panel'):
        heading, count = st.columns([4, 1], vertical_alignment='center')
        threads = retrieve_user_threads(st.session_state['user_id'])
        with heading:
            st.markdown('<div class="history-title">Chat history</div>', unsafe_allow_html=True)
        with count:
            st.markdown(f'<div class="history-count">{len(threads)}</div>', unsafe_allow_html=True)
        st.button('New chat', icon=':material/add:', type='primary', key='history_new_chat',
                  use_container_width=True, on_click=reset_chat)
        search = st.text_input('Search chat history', placeholder='Search chats',
                               label_visibility='collapsed', key='page_history_search')
        with st.container(key='page_thread_list', height=510, border=False):
            matches = 0
            for thread_id in reversed(threads):
                title = get_thread_title(thread_id) or 'New conversation'
                if search.casefold() not in title.casefold():
                    continue
                matches += 1
                st.button(title, key=f'page_thread_{thread_id}', icon=':material/chat_bubble_outline:',
                          type='primary' if thread_id == st.session_state['thread_id'] else 'secondary',
                          use_container_width=True, on_click=_switch_thread, args=(thread_id,))
            if not matches:
                st.caption('No matching conversations.')
        st.caption('Saved in this browser')


def _record(role, content, sources=None):
    storage.append_message(st.session_state['thread_id'], role, content, sources)


def _upload_files(files):
    indexed, failed = [], []
    tid, uid = st.session_state['thread_id'], st.session_state['user_id']
    for file in files:
        with st.status(f'Reading {file.name}', expanded=False) as status:
            try:
                meta = ingest_file(file.getvalue(), tid, file.name, uid)
                indexed.append(meta)
                status.update(label=f"{file.name} is ready", state='complete')
            except Exception as exc:
                logger.error('Document ingestion failed category=%s', type(exc).__name__)
                failed.append((file.name, friendly_error(exc)))
                status.update(label=f'Could not read {file.name}', state='error')
    if indexed:
        names = ', '.join(m['filename'] for m in indexed)
        _record('assistant', f'Added to your document library: {names}. You can now ask questions about these files.')
        if get_thread_title(tid) == 'New conversation':
            set_thread_title(tid, generate_chat_title(indexed[0]['filename']))
    st.session_state['upload_errors'] = failed
    return indexed, failed


def library(meta):
    with st.container(key='library'):
        docs = meta.get('documents', [])
        st.markdown(f'''<div class="library-head"><strong>Document library</strong><span class="count-badge">{len(docs)}</span></div>
          <div class="library-copy">The source of your next lightbulb moment.<br>Add files to this conversation.</div>''', unsafe_allow_html=True)
        files = st.file_uploader('Upload documents', type=SUPPORTED_TYPES, accept_multiple_files=True,
                                 key=f"uploads_{st.session_state['thread_id']}_{st.session_state['uploader_version']}",
                                 label_visibility='collapsed', max_upload_size=25)
        if files and st.button('Add to library', icon=':material/add:', type='primary', use_container_width=True):
            _upload_files(files)
            st.session_state['uploader_version'] += 1
            st.rerun()
        for name, error in st.session_state.pop('upload_errors', []):
            st.error(f'{name}: {error}')
        if not docs:
            st.markdown('''<div class="empty-library"><div class="paper-stack"></div>
              <strong>A fresh page awaits</strong><p>Upload lecture notes, a textbook chapter,<br>or that paper you have been meaning to read.</p></div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label">Ready to explore</div>', unsafe_allow_html=True)
            for doc in docs:
                with st.container(border=True):
                    file_col, remove_col = st.columns([5, 1], vertical_alignment='center')
                    with file_col:
                        st.markdown(f"<div class='file-name'>{html.escape(doc['filename'])}</div><div class='file-meta'>{doc['file_type'].upper()} &nbsp; / &nbsp; {doc['chunks']} passages</div>", unsafe_allow_html=True)
                    with remove_col:
                        if st.button('', icon=':material/close:', key=f"remove_{doc['filename']}", help=f"Remove {doc['filename']}"):
                            remove_thread_document(st.session_state['thread_id'], doc['filename'], st.session_state['user_id'])
                            st.rerun()
        st.markdown('''<div class="library-tip"><b>A small study tip</b><br>Related notes work better together. Add multiple files to compare ideas and connect the dots.</div>''', unsafe_allow_html=True)
        st.caption('PDF, Word, slides, spreadsheets & text. Up to 25 MB per file; 20 files per chat.')


def _render_message(message):
    avatar = ':material/person_outline:' if message['role'] == 'user' else ':material/auto_stories:'
    with st.chat_message(message['role'], avatar=avatar):
        st.markdown(message['content'])
        sources = message.get('sources', [])
        if sources:
            with st.expander(f'View {len(sources)} source passages', icon=':material/library_books:'):
                for source in sources:
                    location = f" / Page {source['page']}" if source.get('page') else ''
                    st.caption(f"[{source['citation']}] {source['source']}{location}")
                    st.text(source.get('text', ''))


def welcome():
    st.markdown('''<div class="hero"><div class="eyebrow">LESS SEARCHING. MORE UNDERSTANDING.</div>
      <h1>A little curiosity.<br><em>A lot of clarity.</em></h1>
      <p>Turn your documents into a conversation. Unpack a tricky concept, find the important details, or prepare for what is next.</p>
      <div class="trust-row"><span>Answers with sources</span><span>Your notes, connected</span><span>Made for learning</span></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="section-label">A good place to start</div>', unsafe_allow_html=True)
    prompts = [
        ('Summarize my notes', 'The key ideas, without the information overload.', 'Summarize the key ideas in my uploaded documents. Cite the source passages.', 'Aa', ''),
        ('Make it make sense', 'Break a complex topic into something clear.', 'Explain the main concept in my uploaded documents in simple terms, with an example.', '?', 'blue'),
        ('Put me to the test', 'Turn your reading into a little active recall.', 'Create five practice questions from my uploaded documents. Put answers at the end and cite the sources.', '5', 'peach'),
    ]
    with st.container(key='prompts'):
        columns = st.columns(3, gap='small')
        for column, (title, desc, prompt, icon, color) in zip(columns, prompts):
            with column, st.container(border=True):
                st.markdown(f'<div class="prompt-icon {color}">{icon}</div><div class="prompt-title">{title}</div><div class="prompt-desc">{desc}</div>', unsafe_allow_html=True)
                st.button('Try this prompt', key=f'prompt_{icon}', use_container_width=True, on_click=_seed_prompt, args=(prompt,))
    st.markdown('<div class="footer-note">Start with a document on the right, or ask a general study question below.<br>Check source passages for details that matter.</div>', unsafe_allow_html=True)


def _answer(question, request_id):
    with st.status('Finding the useful details...' if thread_document_metadata(st.session_state['thread_id']) else 'Thinking through your question...', expanded=False) as status:
        try:
            bot = get_chatbot(st.session_state['user_id'])
            result = bot.invoke({'messages': [HumanMessage(content=question, id=request_id)]},
                                config={'configurable': {'thread_id': st.session_state['thread_id']}, 'recursion_limit': 12})
            final = next((m for m in reversed(result['messages']) if isinstance(m, AIMessage) and not m.tool_calls), None)
            if final is None or not content_text(final.content).strip():
                raise RuntimeError('Empty model response')
            _record('assistant', content_text(final.content), final.additional_kwargs.get('sources', []))
            st.session_state.pop('failed_request', None)
            status.update(label='Answer ready', state='complete')
        except Exception as exc:
            logger.error('Chat failed category=%s', type(exc).__name__)
            st.session_state['failed_request'] = {'question': question, 'id': request_id, 'error': friendly_error(exc)}
            status.update(label='Your question is saved. You can retry below.', state='error')
    st.rerun()


def main_page():
    browser_state = _browser_history_state()
    browser_user_id = valid_user_id(browser_state.get('user_id')) if isinstance(browser_state, dict) else ''
    if (browser_user_id and st.session_state.get('user_id') != browser_user_id
            and not st.session_state.get('browser_history_loaded')):
        st.session_state['user_id'] = browser_user_id
        st.session_state.pop('thread_id', None)
    _init_user_session(browser_user_id)
    _sync_browser_history(browser_state)
    if 'pending_prompt' in st.session_state:
        st.session_state['rag_chat_input'] = st.session_state.pop('pending_prompt')
    st.markdown(_css(), unsafe_allow_html=True)
    sidebar()
    tid = st.session_state['thread_id']
    meta = thread_document_metadata(tid)
    messages = storage.messages(tid)
    status = chat_model_status(st.session_state['user_id'])
    title = get_thread_title(tid) or 'New conversation'
    label = 'Documents connected' if meta else 'Your study space'
    header, settings = st.columns([6, 1.25], gap='small', vertical_alignment='center')
    with header:
        st.markdown(f'''<div class="workspace-bar"><div class="breadcrumb">Workspace &nbsp; / &nbsp; <b>{html.escape(title)}</b></div>
          <div class="status-pill"><span class="status-dot"></span>{label}</div></div>''', unsafe_allow_html=True)
    with settings:
        if st.button('Settings', icon=':material/tune:', use_container_width=True,
                     key='header_settings', type='secondary'):
            settings_dialog()
    with st.container(key='workspace_layout'):
        history, main, rail = st.columns([0.72, 2.45, 1], gap='large')
        with history:
            history_panel()
        with rail:
            library(meta)
        with main:
            if not messages:
                welcome()
            else:
                st.markdown(f'<div class="conversation-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
                document_count = int(meta.get('file_count', 0)) if meta else 0
                passage_count = int(meta.get('chunks', 0)) if meta else 0
                route = str(status.get('provider', 'none')).replace('_', ' ').title()
                st.markdown(f'''<div class="conversation-overview">
                  <div><strong>{document_count}</strong><span>{'document' if document_count == 1 else 'documents'}</span></div>
                  <div><strong>{passage_count}</strong><span>{'passage' if passage_count == 1 else 'passages'} indexed</span></div>
                  <div><strong>{len(messages)}</strong><span>messages</span></div>
                  <div class="route"><span class="status-dot"></span><strong>{html.escape(route)}</strong><span>active route</span></div>
                </div>''', unsafe_allow_html=True)
                with st.container(key='conversation'):
                    for message in messages:
                        _render_message(message)
                controls = st.columns([1.5, 1])
                with controls[0]:
                    st.download_button('Export conversation', export_conversation_txt(messages, title), file_name='teamdino-conversation.txt',
                                       mime='text/plain', icon=':material/download:', use_container_width=True)
                with controls[1]:
                    with st.popover('Manage chat', use_container_width=True):
                        new_title = st.text_input('Conversation name', value=title, max_chars=100)
                        if st.button('Rename conversation', disabled=not new_title.strip()):
                            set_thread_title(tid, new_title.strip())
                            st.rerun()
                        if st.button('Delete this conversation', icon=':material/delete:'):
                            delete_conversation(tid, st.session_state['user_id'])
                            reset_chat()
                            st.rerun()
            if not status['configured']:
                st.info('Add a personal AI key in Settings, or ask an administrator to configure the shared key pool.', icon=':material/key:')
            restored = st.session_state.pop('browser_history_restored', 0)
            if restored:
                st.info(f'Restored {restored} conversation{"s" if restored != 1 else ""} from this browser. Re-upload documents if their server index is no longer available.',
                        icon=':material/history:')
            failed = st.session_state.get('failed_request')
            if failed:
                st.error(failed['error'])
                if st.button('Retry response', icon=':material/refresh:'):
                    _answer(failed['question'], failed['id'])
    submission = st.chat_input('Ask a question, connect an idea, or attach your notes...', key='rag_chat_input',
                               accept_file='multiple', file_type=SUPPORTED_TYPES, max_chars=6000, max_upload_size=25)
    question, files = _coerce_submission(submission)
    if submission is None:
        return
    if files:
        with main:
            _, errors = _upload_files(files)
        if errors:
            # Never answer as if a failed attachment had been read.
            if question:
                st.session_state['pending_prompt'] = question
            st.rerun()
    if question:
        st.session_state.pop('failed_request', None)
        _record('user', question)
        if title == 'New conversation':
            set_thread_title(tid, generate_chat_title(question))
        with main:
            _render_message({'role': 'user', 'content': question})
            _answer(question, str(uuid.uuid4()))
    elif files:
        st.rerun()


if __name__ == '__main__':
    main_page()
