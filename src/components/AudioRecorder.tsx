
import React, { useEffect, useState } from "react";
import { useAudioRecorder } from "react-audio-voice-recorder";
import micIcon from "../../public/mic.svg";
import stopIcon from "../../public/pause_button.svg"
import Waves from "./Waves";
import VerifyPrompt from "./VerifyPrompt";

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

  const [verify,setVerify] = useState({predicted_speaker:"Cloned"});
  const [result, setResult] = useState(false);
  const [myrecording, setMyRecording] = useState(false);

 
// console.log('recordingBlob=>',startRecording)
  useEffect(() => {
    if (recordingBlob) {
      console.log(recordingBlob);
      
      // const url = URL.createObjectURL(recordingBlob);
      // const audio = document.createElement("audio");
      // audio.src = url;
      // audio.controls = true;
      // document.body.appendChild(audio);
    }

    const sendAudioToBackend = async () => {
      console.log('stop Recording')
      if (recordingBlob) {
        setResult(true);
        setMyRecording(true);
        
        const formData = new FormData();
        formData.append('file', recordingBlob, 'recording.wav');
        try {
          console.log('Sending audio to model...');
          const response = await fetch('http://127.0.0.1:8092/predict/', {
            method: 'POST',
            body: formData,
          });
          setVerify(response);
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

  function handleRetry()
  {
    setMyRecording(false);
    setResult(false);
  }

 
  

  return (
    <>
    
      {(!isRecording && !myrecording && !result) &&<button className="start-record" onClick={startRecording} disabled={isRecording}>
        <img src={micIcon} alt="" />
        SPEAK FOR PASSWORD
      </button>}
      
      {(isRecording && !myrecording && !result) && <button onClick={stopRecording} className="stop-record" disabled={!isRecording}>
        <img src={stopIcon} alt="" />
        <Waves />
      </button>}
      {/* <button onClick={togglePauseResume} disabled={!isRecording}>
        {isPaused ? "Resume" : "Pause"}
      </button> */}
      {isRecording && <div>Recording Time: {recordingTime}s</div>}
      {(myrecording && result) && <VerifyPrompt mess={verify} retry={handleRetry}  />}
    </>
  );
};

export default AudioRecorderComponent;

