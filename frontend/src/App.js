import Dashboard from "./Views/Pages/Dashboard";
import Sidenavbar from "./Components/Sidenavbar";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Signin from "./Views/Account Pages/Signin";
import Register from "./Views/Account Pages/Register";
import Profile from "./Views/Account Pages/Profile";
import Tables from "./Views/Pages/Tables";
import Editprofile from "./Views/Account Pages/Editprofile";
import Homepage from "./Views/Pages/Homepage";
import Search from "./Views/Pages/Search";
import Report from "./Views/Pages/Report";
import Monitoring from "./Views/Pages/Monitoring";
import { ProtectedRoute } from "./Views/Account Pages/ProtectedRoute";
import { AuthProvider } from "./context/AuthContext";
import { useContext, useState } from "react";

function App() {
  let [auth, setAuthToken] = useState(
    localStorage.getItem("authToken")
      ? JSON.parse(localStorage.getItem("authToken"))
      : null
  );
  // const auth = JSON.parse(localStorage.getItem("authToken"));
  console.log("auth", auth);
  return (
    <>
      <Router>
        <AuthProvider>
          <Routes>
            <Route exact path="/" element={<Homepage />}></Route>
            <Route path="/signin" element={<Signin />}></Route>
            <Route path="/register" element={<Register />}></Route>
            <Route path="/search" element={<Search />}></Route>
            
            
            <Route element={<ProtectedRoute />}>
              <Route path="/dashboard" element={<Dashboard />}></Route>
            </Route>
            <Route element={<ProtectedRoute />}>
              <Route path="/monitoring" element={<Monitoring />}></Route>
            </Route>
            <Route element={<ProtectedRoute />}>
              <Route path="/profile" element={<Profile />}></Route>
            </Route>
            <Route exact path="/tables" element={<Tables />}></Route>
            <Route exact path="/editprofile" element={<Editprofile />}></Route>
            <Route exact path="/report/:name" element={<Report />}></Route>
            
          </Routes>
        </AuthProvider>
      </Router>
    </>
  );
}

export default App;
