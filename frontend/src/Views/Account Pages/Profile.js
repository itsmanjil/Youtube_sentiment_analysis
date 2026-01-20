import React, { useContext, useEffect, useState, useCallback } from "react";
import Sidenavbar from "../../Components/Sidenavbar";
import Fixedplugins from "../../Components/Fixedplugins";
import { Link } from "react-router-dom";
import jwt_decode from "jwt-decode";
import axios from "axios";
import AuthContext from "../../context/AuthContext";

function Profile(props) {
  const { authToken } = useContext(AuthContext);
  const [user, setUser] = useState({});
  const [searchedList, setSearchedList] = useState([]);
  const getData = useCallback(async () => {
    try {
      const token = localStorage.getItem("authToken");
      const { user_id } = jwt_decode(token);
      console.log(user_id);
      if (user_id) {
        try {
          const userData = await axios({
            method: "GET",
            url: `http://127.0.0.1:8000/api/user/me/${user_id}`,
            timeout: 1000 * 10,
            validateStatus: (status) => {
              return status < 500;
            },
            headers: {
              Authorization: authToken
                ? "Bearer " + String(authToken.access)
                : null,
              "Content-Type": "application/json",
              accept: "application/json",
            },
          });
          setUser({
            user_name: userData.data.user_name,
            email: userData.data.email,
          });
          const list = Array.isArray(userData.data.searched_list)
            ? userData.data.searched_list
            : [];
          setSearchedList(list);
          console.log("user", userData);
        } catch (e) {
          console.log(e);
        }
      }
    } catch (err) {
      console.log(err.message);
    }
  }, [authToken]);
  useEffect(() => {
    getData();
  }, [getData]);
  return (
    <>
      <Sidenavbar />
      <div className="main-content position-relative bg-gray-100 max-height-vh-100 h-100">
        <nav
          className="navbar navbar-main navbar-expand-lg px-0 mx-3 shadow-none border-radius-xl"
          id="navbarBlur"
          data-scroll="true"
        >
          <div className="container-fluid py-1 px-3">
            <nav aria-label="breadcrumb">
              <ol className="breadcrumb bg-transparent mb-0 pb-0 pt-1 px-0 me-sm-6 me-5">
                <li
                  className="breadcrumb-item text-sm text-dark active"
                  aria-current="page"
                ></li>
              </ol>
              <h2 className="font-weight-bolder mb-0">Profile</h2>
            </nav>
            <div
              className="collapse navbar-collapse mt-sm-0 mt-2 me-md-0 me-sm-4"
              id="navbar"
            >
              <div className="ms-md-auto pe-md-3 d-flex align-items-center">
                <div className="input-group input-group-outline">
                  <Link to="/search">
                    <input
                      className="btn btn-light profile-button bg-primary"
                      type="button"
                      value="Analyze Video"
                      style={{
                        margin: 0,
                        textTransform: "capitalize",
                        color: "white",
                      }}
                    ></input>
                  </Link>
                </div>
              </div>
              <ul className="navbar-nav  justify-content-end">
                <li className="nav-item d-flex align-items-center">
                  <span
                    className="nav-link text-body font-weight-bold px-0"
                  >
                    <i className="fa fa-user me-sm-1"></i>
                    <span className="d-sm-inline d-none">{user.user_name}</span>
                  </span>
                </li>
                <li className="nav-item d-xl-none ps-3 d-flex align-items-center">
                  <button
                    type="button"
                    className="nav-link text-body p-0"
                    id="iconNavbarSidenav"
                    style={{background: 'none', border: 'none', cursor: 'pointer'}}
                  >
                    <div className="sidenav-toggler-inner">
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                    </div>
                  </button>
                </li>
                
                
                
                {/* </li> */}
              </ul>
            </div>
          </div>
        </nav>
        <div className="container-fluid px-2 px-md-4">
          <div className="page-header min-height-300 border-radius-xl mt-4">
            {" "}
            {/*style="background-image: url('https://images.unsplash.com/photo-1531512073830-ba890ca4eba2?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');"> */}
            <span className="mask  bg-primary  opacity-6"></span>
          </div>
          <div className="card card-body mx-3 mx-md-4 mt-n6">
            <div className="row gx-4 mb-2">
              <div className="col-auto">
                <div className="avatar avatar-xl position-relative">
                  <img
                    src="../assets/img/bruce-mars.jpg"
                    alt="profile_image"
                    className="w-100 border-radius-lg shadow-sm"
                  />
                </div>
              </div>
              <div className="col-auto my-auto">
                <div className="h-100">
                  <h5 className="mb-1" style={{ textTransform: "uppercase" }}>
                    {user.user_name}
                  </h5>
                  <p className="mb-0 font-weight-normal text-sm">
                    {/* CEO / Co-Founder */}
                    {user.email}
                  </p>
                </div>
              </div>
              <div className="col-lg-4 col-md-6 my-sm-auto ms-sm-auto me-sm-0 mx-auto mt-3">
                <div className="nav-wrapper position-relative end-0">
                  <ul className="nav nav-pills nav-fill p-1" role="tablist">
                    <li className="nav-item">
                      <Link
                        to="/"
                        className="nav-link mb-0 px-0 py-1 active "
                        aria-selected="true"
                      >
                        <i className="material-icons text-lg position-relative">
                          home
                        </i>
                        <span className="ms-1">App</span>
                      </Link>
                    </li>
                    
                    
                  </ul>
                </div>
              </div>
            </div>
            <div className="row">
              <div className="row">
                <div className="col-12 col-xl-4">
                  <div className="card card-plain h-100">
                    <div className="card-header pb-0 p-3">
                      <div className="row">
                        <div className="col-md-8 d-flex align-items-center">
                          <h6 className="mb-0">Profile Information</h6>
                        </div>
                        <div className="col-md-4 text-end">
                          <Link to="/editprofile">
                            <i
                              className="fas fa-user-edit text-secondary text-sm"
                              data-bs-toggle="tooltip"
                              data-bs-placement="top"
                              title="Edit Profile"
                            ></i>
                          </Link>
                        </div>
                      </div>
                    </div>
                    <div className="card-body p-3">
                      <p className="text-sm">
                        Welcome to your profile {user.user_name}.
                      </p>
                      {/* <hr className="horizontal gray-light my-4" /> */}
                      <ul className="list-group">
                        <li className="list-group-item border-0 ps-0 pt-0 text-sm">
                          <strong className="text-dark">Name: </strong> &nbsp;
                          {user.user_name}
                        </li>
                        <li className="list-group-item border-0 ps-0 text-sm">
                          <strong className="text-dark">Email: </strong> &nbsp;
                          {user.email}
                        </li>
                        
                        <li className="list-group-item border-0 ps-0 pb-0">
                          <strong className="text-dark text-sm">Social: </strong>{" "}
                          &nbsp;
                          <a
                            className="btn btn-facebook btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.facebook.com"
                          >
                            <i className="fab fa-facebook fa-lg"></i>
                          </a>
                          <a
                            className="btn btn-reddit btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.reddit.com"
                          >
                            <i className="fab fa-reddit fa-lg"></i>
                          </a>
                          <a
                            className="btn btn-instagram btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.instagram.com"
                          >
                            <i className="fab fa-instagram fa-lg"></i>
                          </a>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                {searchedList.length !== 0 && (
                  <div className="col-12 col-xl-4">
                    <div className="card card-plain h-100">
                      <div className="card-header pb-0 p-3">
                        <h6 className="mb-0">Search History</h6>
                      </div>
                      <div className="card-body p-3">
                        <h6 className="text-uppercase text-body text-xs font-weight-bolder">
                          You searched for:
                        </h6>
                        <ul
                          className="list-group"
                          style={{ maxHeight: "260px", overflowY: "auto" }}
                        >
                          {searchedList.map((item, index) => {
                            const label =
                              typeof item === "string"
                                ? item
                                : item && typeof item === "object"
                                ? item.title || item.video_id
                                : "";
                            const keyBase =
                              item && typeof item === "object"
                                ? item.video_id || item.title || "item"
                                : label || "item";
                            return (
                              <li
                                key={`${keyBase}-${index}`}
                                className="list-group-item border-0 ps-0 pt-0 text-sm"
                              >
                                <div className="text-dark">{label}</div>
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      <Fixedplugins />
    </>
  );
}

export default Profile;
