

// import * as React from 'react';
// import { AudioRecorder } from 'react-audio-voice-recorder';



// export default function AudioRecorderComponent() {
  
  
//   const addAudioElement = (blob: Blob) => {
//     const url = URL.createObjectURL(blob);
//     console.log(blob);
//     const audio = document.createElement('audio');
//     audio.src = url;
//     audio.controls = true;
//     document.body.appendChild(audio);
    
//   };

  

//   return (
//     <div>
//       <AudioRecorder
//         onRecordingComplete={addAudioElement}
//         audioTrackConstraints={{
//           noiseSuppression: true,
//           echoCancellation: true,
//           autoGainControl:true
//           // channelCount,
//           // deviceId,
//           // groupId,
//           // sampleRate,
//           // sampleSize,/ 
//         }}
//         onNotAllowedOrFound={(err) => console.table(err)}
//         downloadOnSavePress={true}
//         downloadFileExtension="webm"
//         mediaRecorderOptions={{
//           audioBitsPerSecond: 128000,
//         }}
//         showVisualizer={true}
//         // onStartRecording={onStart}
       
//       />
//       <br />
//     </div>
//   );
// }

import { log } from "console";
import React, { useEffect } from "react";
import { useAudioRecorder } from "react-audio-voice-recorder";

const AudioRecorderComponent = () => {
  const {
    startRecording,
    stopRecording,
    togglePauseResume,
    recordingBlob,
    isRecording,
    isPaused,
    recordingTime,
  } = useAudioRecorder({
    audioTrackConstraints: {
      noiseSuppression: true,
      echoCancellation: true,
    },
    onNotAllowedOrFound: (error) => {
      console.error("Error accessing audio track:", error);
    },
  });
// console.log('recordingBlob=>',startRecording)
  useEffect(() => {
    if (recordingBlob) {
      console.log(recordingBlob);
      const url = URL.createObjectURL(recordingBlob);
      const audio = document.createElement("audio");
      audio.src = url;
      audio.controls = true;
      document.body.appendChild(audio);
    }

    const sendAudioToBackend = async () => {
      if (recordingBlob) {
        const formData = new FormData();
        formData.append('file', recordingBlob, 'recording.wav');
        try {
          console.log('Sending audio to model...');
          const response = await fetch('http://127.0.0.1:8092/predict/', {
            method: 'POST',
            body: formData,
          });
          // Log response status
          console.log('Response status:', response.status);
          // Check if the response status is not OK
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          const data = await response.json();
          console.log('Response from backend:', data);
        } catch (error) {
          console.error('Error sending audio to backend:', error);
        }
      } else {
        console.log('No blob available to send.');
      }
    };
    sendAudioToBackend();

  }, [recordingBlob]);

  return (
    <div>
      <button onClick={startRecording} disabled={isRecording}>
        Start Recording
      </button>
      <button onClick={stopRecording} disabled={!isRecording}>
        Stop Recording
      </button>
      <button onClick={togglePauseResume} disabled={!isRecording}>
        {isPaused ? "Resume" : "Pause"}
      </button>
      <div>Recording Time: {recordingTime}s</div>
    </div>
  );
};

export default AudioRecorderComponent;

