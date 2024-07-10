import "./ResponseScreen.css";
import threatPic from "../../public/threat.svg"
import roundCheck from "../../public/round-check.svg"

export default function ResponseScreen() {
  return (
    <div className="home">
      <div className="home__inner">
        <h1>Welcome To ThreatEase</h1>
        <div className="app_desc">
          <h1 className="app_desc-name">Application Name 01</h1>
          <p className="app_desc-desc">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </p>
        </div>

        <div className="app_threat">
          <img src={threatPic} alt="Threat Picture" />
          <div className="app_threat-box">
            <h1 className="app_threat-heading">Total Threats Detected : 10</h1>
            <p className="app_threat-desc">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
              eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
              enim ad minim veniam, quis nostrud exercitation
            </p>
          </div>
        </div>

        <div className="app_threat app_threat-green">
          <img src={roundCheck} alt="Threat Picture" />
          <div className="app_threat-box">
            <h1 className="app_threat-heading app_threat-heading-green">Solution</h1>
            <p className="app_threat-desc">
            Sanitize inputs and use parameterized queries to prevent SQL injection.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
