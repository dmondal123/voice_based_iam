import mainLogo from "../../public/logo.svg"

export default function Header()
{
    return(
        <div className="nav">
            <img src={mainLogo} alt="" className="nav__img" />
            
        </div>
    )
}