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
        <h1>Login</h1>
        <div className="login-dialog__par">
          <p>Threat Ease : Your Automated Defense Partner</p>
          <p>Please enter your login details to use the platform.</p>
        </div>

        <div className="login-dialog__prompt">
           <p className="prompt-tag">PROMPT:</p>
           <div className="prompt">
            Hello, my name is [Name]. My voice is my password, secure and safe with ThreatEase.
            </div> 
        </div>
        <AudioRecorderComponent />
        <button type="submit" className={`login ${loginActive? "login--active" : null}`} onClick={handleClick}>LOGIN</button>
      </div>
    </div>
  );
}
