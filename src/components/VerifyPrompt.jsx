import check from "../../public/check.svg"

export default function VerifyPrompt({message})
{
    <div className="verify-prompt">
        <img className="verify-prompt-icon" src={check} alt="" />
        <p className="verify-prompt-message">Password is verified successfully</p>
    </div>
}