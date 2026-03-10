import { ChefHat } from "lucide-react";
import { Link } from "react-router";
const Navbar = () => {
  return (
    <>
      <nav className="sticky top-0 h-16 z-30 bg-background border-b">
        <div className="h-full flex items-center justify-between mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-4">

            <Link to="/" className="text-black-400 text-2xl font-bold">
              <div className="flex gap-2 justify-center items-center">
                <ChefHat />
                Recipe  ChatBot
              </div>
            </Link>
          </div>
        </div>
      </nav>
    </>
  );
};

export default Navbar;
