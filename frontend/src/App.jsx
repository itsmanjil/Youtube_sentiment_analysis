import Dashboard from "./Views/Pages/Dashboard";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
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

function App() {
  return (
    <>
      <Router>
        <AuthProvider>
          <Routes>
            <Route exact path="/" element={<Homepage />}></Route>
            <Route path="/signin" element={<Signin />}></Route>
            <Route path="/register" element={<Register />}></Route>
            <Route element={<ProtectedRoute />}>
              <Route path="/search" element={<Search />}></Route>
              <Route path="/dashboard" element={<Dashboard />}></Route>
              <Route path="/monitoring" element={<Monitoring />}></Route>
              <Route path="/profile" element={<Profile />}></Route>
              <Route exact path="/report/:name" element={<Report />}></Route>
            </Route>
            <Route exact path="/tables" element={<Tables />}></Route>
            <Route exact path="/editprofile" element={<Editprofile />}></Route>
          </Routes>
        </AuthProvider>
      </Router>
    </>
  );
}

export default App;
