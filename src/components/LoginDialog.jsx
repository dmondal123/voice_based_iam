import { useState } from "react";
import AudioRecorderComponent from "./AudioRecorder";


export default function LoginDialog() {

    const [loginActive, setLoginActive] = useState(false);
    function handleActive(newValue)
    {
        setLoginActive(newValue);
    }
    function handleClick()
    {
        setLoginActive(false);
    }



  return (
    <div className="login-dialog">
      <div className="login-dialog__child">
        <h1>Welcome to ThreatEase</h1>
        <div className="login-dialog__par">
          <p>Threat Ease : Your Automated Defense Partner</p>
          <p>Please enter your login details to use the platform.</p>
        </div>

        <div className="login-dialog__username">
           <p>Username</p>
           <input type="text" /> 
        </div>
        <AudioRecorderComponent />
        <button type="submit" className={`login ${loginActive? "login--active" : null}`} onClick={handleClick}>LOGIN</button>
      </div>
    </div>
  );
}
