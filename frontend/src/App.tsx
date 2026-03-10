import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from "react-router";
import HomePage from "./pages/HomePage";
import MainLayout from "./layouts/MainLayout";
import ChatArea from "./pages/ChatArea";

function App() {
  const router = createBrowserRouter(
    createRoutesFromElements(
      <Route path="/" element={<MainLayout />}>
        <Route index element={<HomePage />} />
        <Route path="chat" element={<ChatArea/>}/>
      </Route>,
    ),
  );

  return <RouterProvider router={router} />;
}

export default App;
