// document.getElementById('uploadForm').addEventListener('submit', function(e) {
//     e.preventDefault();

//     const formData = new FormData();
//     const audioFile = document.getElementById('audioFile').files[0];
//     formData.append('audio', audioFile);

//     fetch('/detect/', {
//         method: 'POST',
//         body: formData,
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert(data.error);
//         } else {
//             document.getElementById('result').innerText = `Detected Emotion: ${data.emotion}`;
//         }
//     })
//     .catch(error => console.error('Error:', error));
// });


document.getElementById("audio_file").addEventListener("change", function () {
    alert("Audio file selected!");
});


