import React from "react";
import { Link } from "react-router-dom";
import Navbar from "../../Components/Navbar";
import RegisterForm from "./RegisterForm";
function Register() {
  return (
    <>
    <header className="ex-header">
            <div className="container">
                <div className="row">
                    <div className="col-xl-10 offset-xl-1">
                        <h1 className="text-center">Sign Up</h1>
                    </div> 
                </div> 
            </div> 
      </header>
      <Navbar />
      
      <RegisterForm />
    </>
  );
}

export default Register;
