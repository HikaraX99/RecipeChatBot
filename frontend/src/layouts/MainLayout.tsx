import Navbar from "@/components/Navbar";
import { Outlet } from "react-router";
import { Toaster } from "sonner";

const MainLayout = () => {
  return (
    <>
      <div className="h-dvh flex flex-col overflow-hidden">
        <nav className="fixed top-0 left-0 right-0 h-16 bg-background border-b">
          <Navbar />
        </nav>
        <main className="flex-1 min-h-0 pt-16 overflow-hidden">
          <Outlet />
        </main>
        <Toaster/>
      </div>
    </>
  );
};

export default MainLayout;
