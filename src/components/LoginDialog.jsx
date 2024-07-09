import { useState } from "react";
import AudioRecorderComponent from "./AudioRecorder";
import loginPic from "../../public/login.png";
import { Link , useNavigate, Navigate} from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";



export default function LoginDialog() {

    const [loginActive, setLoginActive] = useState(false);
    const [activeUser , setActiveUser] = useState(localStorage.getItem("activeUser"))
    let history = useNavigate();

    if(activeUser == 'active'){
      return <Navigate to='/home' />
    }

    // function handleActive()
    // {
    //     setLoginActive(true);
        
    // }
    const handleLogin =() =>{
      history('/home')
    }
    



  return (
    <>
    <Header />
    <div className="login-middle">
        <img src={loginPic} alt="" />
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
        <AudioRecorderComponent activateLogin={setLoginActive} />
        <button 
        type="submit" 
        onClick={handleLogin} 
        className={`login ${loginActive?"login--active":null}`} 
        style={{backgroundColor: !loginActive? '#E5E5E5': '#FFB800'}}
        disabled={!loginActive} >
          LOGIN
          </button>
      </div>
    </div>
    </div>
    <Footer />
    </>
  );
}
