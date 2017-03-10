

#include <sarah/cat_problem.h>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>    // std::transform
#include <functional>   // std::plus


/**
 * Main function: Initialise problem and run it.
 */
int main (int argc, char *argv[])
{
  // Initialise MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);

#ifdef RUBBISH
  // Parse the commandline. For this to work we need to compile
  // deal.II with external boost.
  boost::program_options::options_description description {"Options"};
  description.add_options ()
    ("help", "Help screen")
    ("prm", boost::program_options::value<string> ()->default_value ("aphex.prm"), "Parameter file");

 
  boost::program_options::variables_map vmap;
  boost::program_options::store
    (boost::program_options::parse_command_line (argc, argv, description), vmap);
  
  if (vmap.count ("help"))
	std::cout << description
		  << std::endl;
  
  else if (vmap.count ("prm"))
    std::cout << "Parameter file: " << vmap["prm"].as<string>
	      << std::endl;
  
  // else if (...)
#endif
  
  try
    {
      int mpi_comm_rank = 0;
      MPI_Comm_rank (MPI_COMM_WORLD, &mpi_comm_rank);

      int mpi_comm_size = 0;
      MPI_Comm_size (MPI_COMM_WORLD, &mpi_comm_size);

      std::vector<std::string> args (argv+1, argv+argc);
      
      if (mpi_comm_rank==0)
	{
	  AssertThrow (args.size ()>0, dealii::ExcMessage ("The number of input arguments must be greater than zero."));
	  
	  std::cout << std::endl << std::endl
		    << "----------------------------------------------------"
		    << std::endl;
	  std::cout << "Number of processes: " 
		    << mpi_comm_size
		    << std::endl;
	  std::cout << "Caught arguments: ";
	  for (unsigned int i=0; i<args.size (); ++i)
	    std::cout << args[i] << " ";
	  std::cout  << std::endl
		     << "----------------------------------------------------"
		     << std::endl << std::endl;
	}
      
      sarah::CatProblem<2> diabelka (args[0]);
      diabelka.run ();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
