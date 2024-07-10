import React, { useState } from "react";
import Header1 from "./Header1";
import Footer from "./Footer";
import "./Home.css";
import { Flex, Select, Spin } from "antd";
import { getThreat } from "../Api/attack";
import BasicModal from "./Model/Index";
import ResponseScreen from "./ResponseScreen";

export default function Home() {
  // let history = useNavigate();
  // useEffect(() => {
  //   let user = localStorage.getItem("activeUser");
  //   user == null && history("/");
  // }, []);
  // const handleLogout = () => {
  //   console.log("callingg");
  //   localStorage.removeItem("activeUser");
  //   history("/");
  // };

  // WHEN SUBMIT BUTTON IS CALLED

  const [threat, setThreat] = useState("");
  const [spinner, setSpinner] = useState(false);
  const [threatDetected, setThreatDetected] = useState(false);

  const handleSubmit = async (e) => {
    setSpinner(true);
    e.preventDefault();
    let response = await getThreat();
    if (response) {
      // setTimeout(()=>{
      //   setSpinner(false)

      // },5000)
      setSpinner(false);
      setThreat(response);
      // alert("Submit Button Clicked");
    }
  };

  // Handling 'DETECT' button click event
  const handleDetect = async (e) => {
    e.preventDefault();
    setSpinner(true);
    setTimeout(() => {
      setSpinner(false);
      setThreatDetected(true);
    }, 5000);
  };

  const onChange = (value: string) => {
    console.log(`selected ${value}`);
    // setThreat(value);
  };

  const onSearch = (value: string) => {
    console.log("search:", value);
  };

  return (
    <>
      <Header1 />
      {!threatDetected && (
        <div className="home">
          <div className="home__inner">
            <h1>Welcome To ThreatEase</h1>
            <div className="home__threat">
              <form className="home__form">
                <div className="home__input">
                  <Select
                    className="custom-select"
                    showSearch
                    placeholder="Select the Application"
                    optionFilterProp="label"
                    // onChange={onChange}
                    // onSearch={onSearch}
                    options={[
                      {
                        value: "application 1",
                        label: "Application 1",
                      },
                    ]}
                  />
                </div>

                <div className="home__input">
                  <Select
                    className="custom-select"
                    showSearch
                    placeholder="Select the attack"
                    optionFilterProp="label"
                    onChange={onChange}
                    onSearch={onSearch}
                    options={[
                      {
                        value: "SQL Injection",
                        label: "SQL Injection",
                      },
                      {
                        value: "IP Spoofing",
                        label: "IP Spoofing",
                      },
                      {
                        value: "Cross Side Request Forgery",
                        label: "Cross Side Request Forgery",
                      },
                      {
                        value: "Cross Side Scripting",
                        label: "Cross Side Scripting",
                      },
                      {
                        value: "ARP Spoofing",
                        label: "ARP Spoofing",
                      },
                    ]}
                  />
                </div>

                <button
                  type="submit"
                  className="home__submit"
                  onClick={handleSubmit}
                >
                  SUBMIT
                </button>
              </form>
            </div>
            {/* <button onClick={handleLogout}>Logout</button> */}
            {spinner && <BasicModal spinner={spinner} />}
            {threat && (
              <div className="home__detect">
                <h1>Attack was successful!</h1>
                <button
                  type="submit"
                  className="home__submit"
                  onClick={handleDetect}
                >
                  DETECT
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      {threatDetected && <ResponseScreen />}

      <Footer />
    </>
  );
}
