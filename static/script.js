const chatWindow = document.getElementById('chat-window');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');

async function sendPrompt(text) {
    const userMessage = document.createElement('div');
    userMessage.className = 'mb-2 text-right';
    userMessage.innerHTML = `<span class="inline-block p-2 rounded-lg bg-blue-500 text-white">${text}</span>`;
    chatWindow.appendChild(userMessage);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, max_length: 15, temperature: 0.7 })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Response is not valid JSON');
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        const botMessage = document.createElement('div');
        botMessage.className = 'mb-2 text-left';
        botMessage.innerHTML = `<span class="inline-block p-2 rounded-lg bg-gray-200">${data.response}</span>`;
        chatWindow.appendChild(botMessage);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        const errorMessage = document.createElement('div');
        errorMessage.className = 'mb-2 text-left';
        errorMessage.innerHTML = `<span class="inline-block p-2 rounded-lg bg-red-200">Error: ${error.message}</span>`;
        chatWindow.appendChild(errorMessage);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
}

sendBtn.addEventListener('click', () => {
    const text = promptInput.value.trim();
    if (text) {
        sendPrompt(text);
        promptInput.value = '';
    }
});

promptInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});
