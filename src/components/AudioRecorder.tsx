import React, { useEffect, useState } from "react";
import { useAudioRecorder } from "react-audio-voice-recorder";
import micIcon from "../../public/mic.svg";
import stopIcon from "../../public/pause_button.svg";
import Waves from "./Waves";
import VerifyPrompt from "./VerifyPrompt";

const AudioRecorderComponent = ({ activateLogin }) => {
  const {
    startRecording,
    stopRecording,
    recordingBlob,
    isRecording,
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

  const [verify, setVerify] = useState({});
  const [result, setResult] = useState(false);
  const [myRecording, setMyRecording] = useState(false);


  useEffect(() => {
    if (recordingBlob) {
      console.log("Valid Recording Blob: ", recordingBlob);

      const sendAudioToBackend = async () => {
        setResult(true);
        setMyRecording(true);

        

        const formData = new FormData();
        formData.append("file", recordingBlob, "recording.wav");
        try {
          console.log("Sending audio to model...");
          const response = await fetch("http://127.0.0.1:8092/predict/", {
            method: "POST",
            body: formData,
          });
          // Log response status
          console.log("Response status:", response.status);
          // Check if the response status is not OK
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          const data = await response.json();
          console.log("Response from backend:", data);
          setVerify(data);
          if (
            verify.predicted_speaker !== "other" &&
            verify.predicted_speaker !== "Cloned" &&
            verify.predicted_speaker !== "No Voice"
          ) {
            activateLogin();
          }
        } catch (error) {
          console.error("Error sending audio to backend:", error);
        }
      };

      sendAudioToBackend();
    }
  }, [recordingBlob]);

  useEffect(() => {
    console.log("UseEffect called for Recording time => ", recordingTime);
    if (recordingTime === 7) {
      stopRecording();
      console.log(" Recording Blob after 7 secs: ", recordingBlob);
    }
  }, [recordingTime]);

  function handleRetry() {
    setMyRecording(false);
    setResult(false);
    setVerify({});
    
  }

  return (
    <>
      {!isRecording && !myRecording && !result && (
        <button className="start-record" onClick={startRecording} disabled={isRecording}>
          <img src={micIcon} alt="" />
          SPEAK FOR PASSWORD
        </button>
      )}

      {isRecording && !myRecording && !result && (
        <button onClick={stopRecording} className="stop-record" disabled={!isRecording}>
          <img src={stopIcon} alt="" />
          <Waves />
        </button>
      )}
      {isRecording && <div>Recording Time: {recordingTime}s</div>}
      {myRecording && result && <VerifyPrompt mess={verify} retry={handleRetry} />}
    </>
  );
};

export default AudioRecorderComponent;
