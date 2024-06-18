import gridLogo from "../../public/grid-logo.svg"
export default function Footer()
{
    return (
        <div className="footer">
            <div className="footer-logo">
            <img src={gridLogo} alt="" />

            </div>
            <div className="footer-feedback">
            <p className="footer-feedback-help"><span>&#63;</span>Help & Feedback</p>
            <p className="footer-feedback-copy">&copy; Grid Dynamics 2024</p>
            </div>

        </div>

    );
}