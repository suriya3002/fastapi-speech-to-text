let mediaRecorder;
let audioChunks = [];

let startTime;
let timerInterval;


const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

const liveText = document.getElementById("liveText");
const transcription = document.getElementById("transcription");

const timer = document.getElementById("timer");


startBtn.addEventListener("click", async () => {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        mediaRecorder = new MediaRecorder(stream);

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {

            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }

        };

        mediaRecorder.start();

        startBtn.disabled = true;
        stopBtn.disabled = false;

        startTimer();

        liveText.textContent =
            "🎤 Listening...";

    } catch (error) {

        console.error(error);

        alert(
            "Microphone permission is required."
        );

    }

});


stopBtn.addEventListener("click", () => {

    if (!mediaRecorder) {
        return;
    }

    mediaRecorder.stop();

    startBtn.disabled = false;
    stopBtn.disabled = true;

    stopTimer();

});


clearBtn.addEventListener("click", () => {

    liveText.textContent =
        "Your live transcription will appear here...";

    transcription.value = "";

    timer.textContent = "00:00";

});


function startTimer() {

    startTime = Date.now();

    timerInterval = setInterval(() => {

        const elapsed =
            Math.floor(
                (Date.now() - startTime) / 1000
            );

        const minutes =
            String(
                Math.floor(elapsed / 60)
            ).padStart(2, "0");

        const seconds =
            String(
                elapsed % 60
            ).padStart(2, "0");

        timer.textContent =
            `${minutes}:${seconds}`;

    }, 1000);

}


function stopTimer() {

    clearInterval(timerInterval);

}