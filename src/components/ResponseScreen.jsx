import "./ResponseScreen.css";
import threatPic from "../../public/threat.svg"
import roundCheck from "../../public/round-check.svg"
import downloadButton from "../../public/download-button.svg";
import {summary} from "../../logs_test/logs.js"
import DataTable from "./DataTable.jsx";

export default function ResponseScreen({threatData}) {

  // FOR DOWNLOADING THE LOGS FILE

    function handleDownload()
    {
      const handleDownload = async () => {
        try {
          const response = await axios({
            url: 'http://localhost:8080/download', // Adjust the URL to your backend endpoint
            method: 'GET',
            responseType: 'blob', // Important to handle binary data
            params: {
              filePath: '/path/to/your/file.txt' // Adjust the file path parameter as needed
            }
          });
    
          const url = window.URL.createObjectURL(new Blob([response.data]));
          const link = document.createElement('a');
          link.href = url;
          link.setAttribute('download', 'file.txt'); // Adjust the file name as needed
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        } catch (error) {
          console.error('Error downloading the file', error);
        }
      };
    }





  return (
    <div className="response">
      <div className="response__inner">
        <h1>Welcome To ThreatEase</h1>
        <div className="app_desc">
          <h1 className="app_desc-name">WEBGOAT</h1>
          <p className="app_desc-desc">
          WebGoat is an intentionally insecure web application maintained by OWASP, designed to teach web application security lessons through hands-on practice with various vulnerabilities and attack techniques.
          </p>
        </div>

        <div className="app_threat">
          <img src={threatPic} alt="Threat Picture" />
          <div className="app_threat-box">
            <h1 className="app_threat-heading">Total Suspicious Logs : 10</h1>
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

        <div className="terminal">
          <div className="terminal__top">
            <p>Terminal</p>
            <button onClick={handleDownload}><img src={downloadButton} alt="" /></button>
          </div>
          <div className="terminal__bottom">
            {summary["logs_with_attacks"].map((el,idx)=>{
              return (<p key={idx}>{el}</p>);
            })}
          </div>
        </div>

        <DataTable />
      </div>
    </div>
  );
}
