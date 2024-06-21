import Header from "./components/Header";
import LoginDialog from "./components/LoginDialog";
import Footer from "./components/Footer";
import loginPic from "../public/login.png";


export default function App() {
  return (
    <>
      <Header />
      <div className="login-middle">
        <img src={loginPic} alt="" />
        <LoginDialog />
      </div>

      <Footer />
    </>
    
  );
}
