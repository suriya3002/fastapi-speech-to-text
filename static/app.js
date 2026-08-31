document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const audioInput = document.getElementById('audio-input');
    const filePanel = document.getElementById('file-panel');
    const fileName = document.getElementById('file-name');
    const fileMeta = document.getElementById('file-meta');
    const audioPreview = document.getElementById('audio-preview');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    const languageSelect = document.getElementById('language-select');
    const btnConvert = document.getElementById('btn-convert');

    // State Sections
    const sectionUpload = document.getElementById('section-upload');
    const sectionProcessing = document.getElementById('section-processing');
    const sectionResult = document.getElementById('section-result');
    const processingStatusText = document.getElementById('processing-status-text');

    // Result Elements
    const resultFilenameLabel = document.getElementById('result-filename-label');
    const btnDownload = document.getElementById('btn-download');
    const btnCopy = document.getElementById('btn-copy');
    const copyBtnText = document.getElementById('copy-btn-text');
    const btnReset = document.getElementById('btn-reset');
    const textOutput = document.getElementById('text-output');
    const statWords = document.getElementById('stat-words');
    const statChars = document.getElementById('stat-chars');
    const statLang = document.getElementById('stat-lang');
    const statTime = document.getElementById('stat-time');

    let currentFile = null;
    const MAX_SIZE_MB = 200;

    // Helper: format bytes
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Handle File Selection
    function handleFile(file) {
        if (!file) return;

        // Size check (200MB)
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            alert(`File is too large (${formatBytes(file.size)}). Maximum supported file size is ${MAX_SIZE_MB}MB.`);
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        fileMeta.textContent = `${formatBytes(file.size)} • ${file.type || 'Audio file'}`;

        // Create audio object URL for preview
        const fileUrl = URL.createObjectURL(file);
        audioPreview.src = fileUrl;

        // UI toggles
        filePanel.classList.remove('hidden');
        dropzone.classList.add('hidden');
        btnConvert.disabled = false;
    }

    // Reset Selected File
    function clearSelectedFile() {
        currentFile = null;
        audioInput.value = '';
        if (audioPreview.src) {
            URL.revokeObjectURL(audioPreview.src);
            audioPreview.src = '';
        }
        filePanel.classList.add('hidden');
        dropzone.classList.remove('hidden');
        btnConvert.disabled = true;
    }

    // Event Listeners for File Selection
    audioInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files[0]) {
            handleFile(dt.files[0]);
        }
    });

    // Conversion Action
    btnConvert.addEventListener('click', async () => {
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('language', languageSelect ? languageSelect.value : 'en');

        // Show Processing Screen
        sectionUpload.classList.add('hidden');
        sectionResult.classList.add('hidden');
        sectionProcessing.classList.remove('hidden');
        processingStatusText.textContent = 'Transcribing English audio with ultra-fast Whisper AI...';

        try {
            const response = await fetch('/convert', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let errMsg = 'Failed to convert audio file.';
                try {
                    const errData = await response.json();
                    if (errData.detail) errMsg = errData.detail;
                } catch (_) {}
                throw new Error(errMsg);
            }

            const result = await response.json();

            // Populate Result
            textOutput.value = result.text || '';
            resultFilenameLabel.textContent = result.txt_filename || 'transcription.txt';
            btnDownload.href = result.download_url || '#';
            btnDownload.setAttribute('download', result.txt_filename || 'transcription.txt');

            statWords.textContent = result.stats?.word_count ?? 0;
            statChars.textContent = result.stats?.char_count ?? 0;
            statLang.textContent = result.stats?.detected_language ?? 'ENGLISH (EN)';
            statTime.textContent = (result.stats?.process_time_sec ?? 0) + 's';

            // Show Result Screen
            sectionProcessing.classList.add('hidden');
            sectionResult.classList.remove('hidden');

        } catch (error) {
            alert('Error: ' + error.message);
            sectionProcessing.classList.add('hidden');
            sectionUpload.classList.remove('hidden');
        }
    });

    // Copy to Clipboard
    btnCopy.addEventListener('click', async () => {
        if (!textOutput.value) return;

        try {
            await navigator.clipboard.writeText(textOutput.value);
            copyBtnText.textContent = 'Copied!';
            btnCopy.classList.add('btn-primary');
            btnCopy.classList.remove('btn-secondary');

            setTimeout(() => {
                copyBtnText.textContent = 'Copy Text';
                btnCopy.classList.remove('btn-primary');
                btnCopy.classList.add('btn-secondary');
            }, 2000);
        } catch (err) {
            textOutput.select();
            document.execCommand('copy');
            copyBtnText.textContent = 'Copied!';
            setTimeout(() => { copyBtnText.textContent = 'Copy Text'; }, 2000);
        }
    });

    // Reset to Convert Another
    btnReset.addEventListener('click', () => {
        clearSelectedFile();
        textOutput.value = '';
        sectionResult.classList.add('hidden');
        sectionUpload.classList.remove('hidden');
    });
});