import Navbar from "../../Components/Navbar";
import RegisterForm from "./RegisterForm";
// Landing-theme styles (.ex-header, .ex-form-1, .offcanvas-collapse): needed
// for direct loads of /register, which otherwise render an unstyled navbar.
import "../Pages/Homepage.css";
import usePageTitle from "../../utils/usePageTitle";
function Register() {
  usePageTitle("Sign Up");
  return (
    <>
      <Navbar />
      <header className="ex-header">
            <div className="container">
                <div className="row">
                    <div className="col-xl-10 offset-xl-1">
                        <h1 className="text-center">Sign Up</h1>
                    </div>
                </div>
            </div>
      </header>
      <RegisterForm />
    </>
  );
}

export default Register;
