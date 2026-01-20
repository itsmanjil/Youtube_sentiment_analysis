import { React, useState, useEffect, useContext } from "react";
import { Link } from "react-router-dom";
import AuthContext from "../../context/AuthContext";

const SigninForm = () => {
  let { loginUser } = useContext(AuthContext);

  const [inputData, setInputData] = useState({
    email: "",
    password: "",
  });

  const { email, password } = inputData;

  const { isError } = useContext(AuthContext);

  const inputHanlder = (e) => {
    setInputData({
      ...inputData,
      [e.target.name]: e.target.value,
    });
  };

  const submitHandler = async (e) => {
    e.preventDefault();
    loginUser(email, password);
  };

  return (
    <>
      <div className="ex-form-1 pt-5 pb-5">
        <div className="container">
          <div className="row">
            <div className="col-xl-6 offset-xl-3">
              <div className="text-box mt-5 mb-5">
                <p className="mb-4">
                  Don't have an account? Then please{" "}
                  <Link to="/register">
                    <span style={{ color: "#ff0000" }}>Sign up</span>
                  </Link>
                </p>

                <form role="form" className="text-start">
                  {isError && (
                    <p style={{ color: "red" }}>
                      Invalid username or password.
                    </p>
                  )}
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
                  <div className="input-group input-group-outline mb-3">
                    {/* <label className="form-label">Password</label> */}
                    <input
                      type="password"
                      placeholder="Password"
                      className="form-control"
                      name="password"
                      value={password}
                      onChange={inputHanlder}
                    />
                  </div>
                  
                  <div className="text-center">
                    <button
                      type="submit"
                      className="p-2 mb-2 bg-primary text-white w-100 my-4 mb-2"
                      // className="btn btn-primary w-100 my-4 mb-2"
                      onClick={submitHandler}
                    >
                      Log In
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

export default SigninForm;
