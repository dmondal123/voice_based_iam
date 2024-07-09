import mainLogo from "../../public/logo.svg";
import { useContext } from "react";
import { userData } from "../Context";
import info from "../Data";
import "./Header1.css";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Header() {

  // LOGOUT FUCTIONALITY
   let history = useNavigate();
  useEffect(() => {
    let user = localStorage.getItem("activeUser");
    user == null && history("/");
  }, []);
  const handleLogout = () => {
    console.log("callingg");
    localStorage.removeItem("activeUser");
    history("/");
  };






  function myUser(firstName) {
    const result = info.filter((person) => person.fullName.includes(firstName));
    // return result;
    return result.map((person) => person.fullName);
  }

  function getInitials(name) {
    // Split the name into an array of words
    const nameArray = name.trim().split(" ");

    // Extract the first letter of each word and join them
    const initials = nameArray
      .map((word) => word.charAt(0).toUpperCase())
      .join("");
    return initials;
  }

  function getSlackID(fname) {
    const result = info.filter((person) => person.fullName.includes(fname));
    // return result;
    return result.map((person) => person.slack);
  }

  function getRole(fname) {
    const result = info.filter((person) => person.fullName.includes(fname));
    // return result;
    return result.map((person) => person.role);
  }

  function getMail(fname) {
    const result = info.filter((person) => person.fullName.includes(fname));
    // return result;
    return result.map((person) => person.mail);
  }

  // Example usage, used with Context API and verify state from AudioRecorder

  // const {userName} = useContext(userData);
  // const fullname = firstLastName(userName);
  // const initials = getInitials(fullname) ;

  const userLoggedIn = myUser(localStorage.getItem("activeUserName"));
  const fullname = userLoggedIn[0];
  console.log("Full Name of User : ", fullname);
  const initials = getInitials(fullname);
  console.log("User's Initials : ", initials);
  const [slackID] = getSlackID(fullname);
  console.log("Slack ID : ", slackID);

  const [role] = getRole(fullname);
  console.log("Rolw : ", role);
  const [mail] = getMail(fullname);
  console.log("Email Address : ", mail);

  return (
    <div className="nav">
      <img src={mainLogo} alt="" className="nav__img" />
      <div className="nav__user">
        <p>
          <span className="nav__initials">{initials}</span>
        </p>
        <p className="nav__fullname">{fullname}</p>
        <div className="dropdown">
          <button className="dropdown-button"> ▼</button>
          <div className="dropdown-content">
            <a href="#"><i className="fa-brands fa-slack dropdowm__icon"></i>{slackID}</a>
            <a href="#"><i className="fa-solid fa-envelope dropdowm__icon"></i>{mail}</a>
            <a href="#"><i className="fa-solid fa-graduation-cap dropdowm__icon"></i>{role}</a>
            <a href="#"><button onClick={handleLogout} className="logout">LOG OUT</button></a>
          </div>
        </div>
      </div>
    </div>
  );
}
