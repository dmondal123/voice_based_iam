import check from "../../public/check.svg"
import error from "../../public/error.svg"

export default function VerifyPrompt({mess, retry})
{

    // function handleretry()
    // {
    //     retry();
    // }
    let text="";
    if(mess.predicted_speaker === "Cloned")
        {
            text = (<div className="verify-prompt verify-prompt-wrong">
            <img className="verify-prompt-icon" src={error} alt="" />
            <p className="verify-prompt-message">Cloned Voice Detected... </p>
            <button className="verify-prompt-wrong-retry" onClick={retry}>RETRY</button>
        </div>);
        }
        else if(mess.predicted_speaker === "other")
            {
                text = (<div className="verify-prompt verify-prompt-wrong">
                    <img className="verify-prompt-icon" src={error} alt="" />
                    <p className="verify-prompt-message">User not in Database... </p>
                    <button className="verify-prompt-wrong-retry" onClick={retry}>RETRY</button>
                </div>);
            }
        else if(mess.predicted_speaker === "No Voice")
            {
                text = (<div className="verify-prompt verify-prompt-wrong">
                    <img className="verify-prompt-icon" src={error} alt="" />
                    <p className="verify-prompt-message">No voice input detected... </p>
                    <button className="verify-prompt-wrong-retry" onClick={retry}>RETRY</button>
                </div>);
            }
            else
            {
                text = (
                        <div className="verify-prompt verify-prompt-right">
            <img className="verify-prompt-icon" src={check} alt="" />
            <p className="verify-prompt-message">User {mess.predicted_speaker} is verified successfully...</p>
        </div>
                );
            }
    return text;
   
      
    
    
    
}