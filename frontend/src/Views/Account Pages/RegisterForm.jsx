import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axiosInstance from "../../axios";

const RegisterForm = () => {
  const navigate = useNavigate();
  const [inputData, setInputData] = useState({
    user_name: "",
    email: "",
    password: "",
    password2: "",
  });

  const [formErrors, setFormErrors] = useState([]);
  const [emailError, setEmailError] = useState({});
  const [usernameError, setUsernameError] = useState({});

  const { user_name, email, password, password2 } = inputData;

  const inputHanlder = (e) => {
    setInputData({
      ...inputData,
      [e.target.name]: e.target.value,
      error: false,
    });
  };

  const validationHanlder = (inputData) => {
    const error = {};
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;
    const nameRegex = /^[A-Za-z\s]+$/;
    if (!inputData.user_name) {
      error.user_name = "Name is required!";
    }
    if (!nameRegex.test(inputData.user_name)) {
      error.user_name = "Valid name is required!";
    }
    if (!inputData.email) {
      error.email = "Email is required";
    } else if (!regex.test(inputData.email)) {
      error.email = "This is not a valid email format!";
    }
    if (!inputData.password) {
      error.password = "Both password are required";
    } else if (inputData.password.length < 4) {
      error.password = "Password must be more than 4 characters";
    } else if (inputData.password.length > 10) {
      error.password = "Password cannot exceed more than 10 characters";
    }
    if (!inputData.password2) {
      error.password2 = "Both password are required";
    } else if (inputData.password2.length < 4) {
      error.password2 = "Password must be more than 4 characters";
    } else if (inputData.password2.length > 10) {
      error.password2 = "Password cannot exceed more than 10 characters";
    }
    if (inputData.password !== inputData.password2) {
      error.password2 = "Password doesn't match!";
    }
    return error;
  };

  const storeDataHandler = () => {
    axiosInstance
      .post("user/register/", {
        user_name: user_name,
        email: email,
        password: password,
        password2: password2,
      })
      .then((res) => {
        if (res.status === 400) {
          const email = {};
          const user_name = {};
          email.error = res.data.email;
          user_name.error = res.data.user_name;
          setEmailError(email);
          setUsernameError(user_name);
        } else {
          alert("User created");
          navigate("/signin");
        }
      })
      .catch(() => {});
  };

  const submitHandler = (e) => {
    e.preventDefault();
    let errorForm = validationHanlder(inputData);
    setFormErrors(errorForm);
    if (Object.keys(errorForm).length === 0) {
      storeDataHandler();
    }
  };

  return (
    <>
      <div className="ex-form-1 pt-5 pb-5">
        <div className="container">
          <div className="row">
            <div className="col-xl-6 offset-xl-3">
              <div className="text-box mt-5 mb-5">
                <p className="mb-4">
                  Fill out the form below to sign up for the service. Already
                  signed up? Then just{" "}
                  <Link to="/signin">
                    <span style={{ color: "#ff0000" }}>Sign in</span>
                  </Link>
                </p>
                <form role="form">
                  <div className="input-group input-group-outline mb-3">
                    {/* <label className="form-label">Name</label> */}
                    <input
                      type="text"
                      placeholder="Name"
                      className="form-control"
                      name="user_name"
                      value={user_name}
                      onChange={inputHanlder}
                    />
                  </div>
                  <p style={{ color: "red" }}>{formErrors.user_name}</p>
                  <p style={{ color: "red" }}>{usernameError.error}</p>
                  <div className="input-group input-group-outline mb-3">
                    {/* <label className="form-label">Email</label> */}
                    <input
                      type="email"
                      placeholder="Email"
                      className="form-control"
                      name="email"
                      value={email}
                      onChange={inputHanlder}
                    />
                  </div>
                  <p style={{ color: "red" }}>{formErrors.email}</p>
                  <p style={{ color: "red" }}>{emailError.error}</p>
                  <div className="input-group input-group-outline mb-3">
                    <input
                      type="password"
                      placeholder="Password"
                      className="form-control"
                      name="password"
                      value={password}
                      onChange={inputHanlder}
                    />
                  </div>
                  <p style={{ color: "red" }}>{formErrors.password}</p>
                  <div className="input-group input-group-outline mb-3">
                    <input
                      type="password"
                      placeholder="Confirm Password"
                      className="form-control"
                      name="password2"
                      value={password2}
                      onChange={inputHanlder}
                    />
                  </div>
                  <p style={{ color: "red" }}>{formErrors.password2}</p>
                  
                  <div className="text-center">
                    <button
                      type="submit"
                      className="p-2 mb-2 bg-primary text-white w-100 my-4 mb-2"
                      onClick={submitHandler}>
                      Sign Up
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default RegisterForm;
