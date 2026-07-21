import { useState } from "react";
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
  // Keyed by field name (user_name/email/password/password2) — holds
  // whatever per-field validation errors the backend returns (DRF serializer
  // errors are arrays of strings per field).
  const [serverErrors, setServerErrors] = useState({});
  const [submitError, setSubmitError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const serverFieldError = (field) => {
    const value = serverErrors[field];
    if (!value) return null;
    return Array.isArray(value) ? value.join(" ") : value;
  };

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
    } else if (!nameRegex.test(inputData.user_name)) {
      error.user_name = "Valid name is required!";
    }
    if (!inputData.email) {
      error.email = "Email is required";
    } else if (!regex.test(inputData.email)) {
      error.email = "This is not a valid email format!";
    }
    if (!inputData.password) {
      error.password = "Password is required";
    } else if (inputData.password.length < 8) {
      error.password = "Password must be at least 8 characters";
    } else if (inputData.password.length > 128) {
      error.password = "Password cannot exceed 128 characters";
    }
    if (!inputData.password2) {
      error.password2 = "Please confirm your password";
    } else if (inputData.password2.length < 8) {
      error.password2 = "Password must be at least 8 characters";
    } else if (inputData.password2.length > 128) {
      error.password2 = "Password cannot exceed 128 characters";
    }
    if (inputData.password !== inputData.password2) {
      error.password2 = "Password doesn't match!";
    }
    return error;
  };

  const storeDataHandler = () => {
    setSubmitError("");
    setServerErrors({});
    setIsSubmitting(true);
    axiosInstance
      .post("user/register/", {
        user_name: user_name,
        email: email,
        password: password,
        password2: password2,
      })
      .then((res) => {
        if (res.status === 400) {
          setServerErrors(res.data || {});
          setIsSubmitting(false);
        } else if (res.status >= 200 && res.status < 300) {
          navigate("/signin", { state: { justRegistered: true } });
        } else {
          setSubmitError(res.data?.message || res.data?.msg || "Registration failed. Please try again.");
          setIsSubmitting(false);
        }
      })
      .catch(() => {
        setSubmitError("Cannot connect to server. Please check your connection and try again.");
        setIsSubmitting(false);
      });
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
                  {submitError && (
                    <div className="alert alert-danger" role="alert">{submitError}</div>
                  )}
                  <div className="input-group input-group-outline mb-3">
                    {/* <label className="form-label">Name</label> */}
                    <input
                      type="text"
                      placeholder="Name"
                      aria-label="Name"
                      className="form-control"
                      name="user_name"
                      value={user_name}
                      onChange={inputHanlder}
                    />
                  </div>
                  {(formErrors.user_name || serverFieldError("user_name")) && (
                    <p className="text-danger text-sm mb-3">
                      <i className="fas fa-exclamation-circle me-1" aria-hidden="true"></i>
                      {formErrors.user_name || serverFieldError("user_name")}
                    </p>
                  )}
                  <div className="input-group input-group-outline mb-3">
                    {/* <label className="form-label">Email</label> */}
                    <input
                      type="email"
                      placeholder="Email"
                      aria-label="Email"
                      className="form-control"
                      name="email"
                      value={email}
                      onChange={inputHanlder}
                    />
                  </div>
                  {(formErrors.email || serverFieldError("email")) && (
                    <p className="text-danger text-sm mb-3">
                      <i className="fas fa-exclamation-circle me-1" aria-hidden="true"></i>
                      {formErrors.email || serverFieldError("email")}
                    </p>
                  )}
                  <div className="input-group input-group-outline mb-3">
                    <input
                      type="password"
                      placeholder="Password"
                      aria-label="Password"
                      className="form-control"
                      name="password"
                      value={password}
                      onChange={inputHanlder}
                    />
                  </div>
                  {(formErrors.password || serverFieldError("password")) && (
                    <p className="text-danger text-sm mb-3">
                      <i className="fas fa-exclamation-circle me-1" aria-hidden="true"></i>
                      {formErrors.password || serverFieldError("password")}
                    </p>
                  )}
                  <div className="input-group input-group-outline mb-3">
                    <input
                      type="password"
                      placeholder="Confirm Password"
                      aria-label="Confirm Password"
                      className="form-control"
                      name="password2"
                      value={password2}
                      onChange={inputHanlder}
                    />
                  </div>
                  {(formErrors.password2 || serverFieldError("password2")) && (
                    <p className="text-danger text-sm mb-3">
                      <i className="fas fa-exclamation-circle me-1" aria-hidden="true"></i>
                      {formErrors.password2 || serverFieldError("password2")}
                    </p>
                  )}

                  <div className="text-center">
                    <button
                      type="submit"
                      className="p-2 mb-2 bg-primary text-white w-100 my-4 mb-2"
                      onClick={submitHandler}
                      disabled={isSubmitting}
                    >
                      {isSubmitting ? "Signing up..." : "Sign Up"}
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
