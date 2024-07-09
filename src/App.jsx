import LoginDialog from "./components/LoginDialog";
import { BrowserRouter, Route, Link, Routes, Navigate } from "react-router-dom";
import { useState } from "react";
import Home from "./components/Home";
import { userData } from "./Context";

// //Spinner test
// import { Flex, Spin } from 'antd';


export default function App() {
  const [activeUer, setActiveUser] = useState(
    localStorage.getItem("activeUser")
  );
  const [userName, setUserName] = useState();
  return (
    <>
    <userData.Provider value={{userName, setUserName}}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LoginDialog />}></Route>
          <Route path="/home" element={<Home/>}></Route>
        </Routes>
      </BrowserRouter>
      </userData.Provider>
    </>
  );
}
