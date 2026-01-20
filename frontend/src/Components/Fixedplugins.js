import React from 'react'

function Fixedplugins() {
    return (
        <>
            <div className="fixed-plugin">
                
                <div className ="card shadow-lg">
                <div className ="card-header pb-0 pt-3">
                <div className ="float-start">
                <h5 className ="mt-3 mb-0">Material UI Configurator</h5>
                <p>See our dashboard options.</p>
                </div>
                <div className ="float-end mt-4">
                <button className ="btn btn-link text-dark p-0 fixed-plugin-close-button">
                <i className ="material-icons">clear</i>
                </button>
                </div>
                {/* <!--End Toggle Button--> */}
                </div>
                <hr className ="horizontal dark my-1" />
                <div className ="card-body pt-sm-3 pt-0">
                {/* <!--Sidebar Backgrounds--> */}
                <div>
                <h6 className ="mb-0">Sidebar Colors</h6>
                </div>
                <button type="button" className ="switch-trigger background-color" style={{background: 'none', border: 'none', padding: 0, cursor: 'pointer'}}>
                <div className ="badge-colors my-2 text-start">
                <span
                  className="badge filter bg-gradient-primary active"
                  data-color="primary"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                <span
                  className="badge filter bg-gradient-dark"
                  data-color="dark"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                <span
                  className="badge filter bg-gradient-info"
                  data-color="info"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                <span
                  className="badge filter bg-gradient-success"
                  data-color="success"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                <span
                  className="badge filter bg-gradient-warning"
                  data-color="warning"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                <span
                  className="badge filter bg-gradient-danger"
                  data-color="danger"
                  onClick={(event) => window.sidebarColor?.(event.currentTarget)}
                ></span>
                </div>
                </button>
                {/* <!--Sidenav Type--> */}
                <div className ="mt-3">
                <h6 className ="mb-0">Sidenav Type</h6>
                <p className ="text-sm">Choose between 2 different sidenav types.</p>
                </div>
                <div className ="d-flex">
                <button
                  className="btn bg-gradient-dark px-3 mb-2 active"
                  data-class="bg-gradient-dark"
                  onClick={(event) => window.sidebarType?.(event.currentTarget)}
                >
                  Dark
                </button>
                <button
                  className="btn bg-gradient-dark px-3 mb-2 ms-2"
                  data-class="bg-transparent"
                  onClick={(event) => window.sidebarType?.(event.currentTarget)}
                >
                  Transparent
                </button>
                <button
                  className="btn bg-gradient-dark px-3 mb-2 ms-2"
                  data-class="bg-white"
                  onClick={(event) => window.sidebarType?.(event.currentTarget)}
                >
                  White
                </button>
                </div>
                <p className ="text-sm d-xl-none d-block mt-2">You can change the sidenav type just on desktop view.</p>
                {/* <!--Navbar Fixed--> */}
                <div className ="mt-3 d-flex">
                <h6 className ="mb-0">Navbar Fixed</h6>
                <div className ="form-check form-switch ps-0 ms-auto my-auto">
                <input
                  className="form-check-input mt-1 ms-auto"
                  type="checkbox"
                  id="navbarFixed"
                  onClick={(event) => window.navbarFixed?.(event.currentTarget)}
                />
                </div>
                </div>
                <hr className ="horizontal dark my-3" />
                <div className ="mt-2 d-flex">
                <h6 className ="mb-0">Light / Dark</h6>
                <div className ="form-check form-switch ps-0 ms-auto my-auto">
                <input
                  className="form-check-input mt-1 ms-auto"
                  type="checkbox"
                  id="dark-version"
                  onClick={(event) => window.darkMode?.(event.currentTarget)}
                />
                </div>
                </div>
                <hr className ="horizontal dark my-sm-4" />
                <a className ="btn btn-outline-dark w-100" href="#documentation">View documentation</a>
                <div className ="w-100 text-center">
                <a className ="github-button" href="https://github.com/creativetimofficial/material-dashboard" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star creativetimofficial/material-dashboard on GitHub">Star</a>
                <h6 className ="mt-3">Thank you for sharing!</h6>
                <a href="https://www.reddit.com/submit?url=https%3A%2F%2Fwww.creative-tim.com%2Fproduct%2Fmaterial-dashboard&title=Check%20Material%20UI%20Dashboard" className ="btn btn-dark mb-0 me-2" target="_blank" rel="noreferrer">
                <i className ="fab fa-reddit me-1" aria-hidden="true"></i> Share on Reddit
                </a>
                <a href="https://www.facebook.com/sharer/sharer.php?u=https://www.creative-tim.com/product/material-dashboard" className ="btn btn-dark mb-0 me-2" target="_blank" rel="noreferrer">
                <i className ="fab fa-facebook-square me-1" aria-hidden="true"></i> Share
                </a>
                </div>
                </div>
                </div>
                </div>
        </>
    )
}

export default Fixedplugins
