
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <sarah/matrix_creator.h>

#include <fstream>
#include <iostream>

#include <algorithm>    // std::transform
#include <functional>   // std::plus

namespace sarah
{

  /**
   * Solve the system Mx=b, where M is the mass matrix and b is a
   * linear function.
   */
  template <int dim>
  class CatProblem
  {
  public:

    /**
     * Class constructor.
     */
    CatProblem (const std::string &prm);

    /**
     * Class destructor.
     */
    ~CatProblem ();

    /**
     * Wrapper function, that controls the order of excecution.
     */
    void run ();
    
  private:

    /**
     * Make intial coarse grid.
     */
    void make_coarse_grid ();

    /**
     * Setup system matrices and vectors.
     */
    void setup_system ();

    /**
     * Assemble system matrices and vectors.
     */
    void assemble_system();

    /**
     * Solve the linear algebra system.
     */
    unsigned int solve ();
    
    /**
     * Output results, ie., finite element functions and derived
     * quantitites for this cycle.
     */
    void output_results (const unsigned int cycle);

    /**
     * Refine grid based on Kelly's error estimator working on the
     * material id (solution vector).
     */
    void refine_grid ();
    
    /**
     * MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /**
     * A distributed grid on which all computations are done.
     */
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    /**
     * Scalar DoF handler primarily used for interpolating material
     * identification.
     */
    dealii::DoFHandler<dim> dof_handler;

    /**
     * Scalar valued finite element primarily used for interpolating
     * material iudentification.
     */
    dealii::FESystem<dim> fe;
    
    /**
     * Index set of locally owned DoFs.
     */
    dealii::IndexSet locally_owned_dofs;
    
    /**
     * Index set of locally relevant DoFs.
     */
    dealii::IndexSet locally_relevant_dofs;
    
    /**
     * A list of (hanging node) constraints.
     */
    dealii::ConstraintMatrix constraints;
    
    /**
     * System matrix.
     */
    dealii::PETScWrappers::MPI::SparseMatrix system_matrix;

    /**
     * Mass matrix.
     */
    dealii::PETScWrappers::MPI::SparseMatrix mass_matrix;

    /**
     * Locally relevant solution vector.
     */
    std::vector<dealii::PETScWrappers::MPI::Vector> locally_relevant_solution;

    /**
     * Solution value.
     */
    std::vector<double> solution_value;

    /**
     * Number of eigenpairs to solve for.
     */
    const unsigned int n_pairs;
    
    /**
     * Parallel iostream.
     */
    dealii::ConditionalOStream pcout;

    /**
     * Stop clock.
     */
    dealii::TimerOutput timer;
    
    /**
     * Input parameter file.
     */
    dealii::ParameterHandler parameters;
    
  }; // CatProblem

  
  /**
   * Class constructor.
   */
  template <int dim>
  CatProblem<dim>::CatProblem (const std::string &prm)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename dealii::Triangulation<dim>::MeshSmoothing
                   (dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening)),
    dof_handler (triangulation),
    fe (dealii::FE_Q<dim> (3), 1),
    n_pairs (3),
    // ---
    pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process (mpi_communicator) == 0)),
    timer (mpi_communicator, pcout,
	   dealii::TimerOutput::summary,
	   dealii::TimerOutput::wall_times)
  {
    parameters.declare_entry ("Global mesh refinement steps", "5",
                              dealii::Patterns::Integer (0, 20),
                              "The number of times the 1-cell coarse mesh should "
                              "be refined globally for our computations.");

    parameters.declare_entry ("Potential", "0",
                              dealii::Patterns::Anything (),
                              "A functional description of the potential.");
    
    parameters.declare_entry ("Number of eigenpairs", "5",
                              dealii::Patterns::Integer (0, 100),
                              "The number of eigenpairs to be computed.");

    parameters.declare_entry ("Number of eigenfunctions", "5",
                              dealii::Patterns::Integer (0, 100),
                              "The number of eigenfunctions "
                              "to be used for error indicators.");
    
    parameters.parse_input (prm);
  }

  
  /**
   * Class destructor.
   */
  template <int dim>
  CatProblem<dim>::~CatProblem ()
  {
    // Wipe DoF handlers.
    dof_handler.clear ();
  }


  /**
   * Make initial coarse grid.
   */
  template <int dim>
  void
  CatProblem<dim>::make_coarse_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "make coarse grid");

    // Create a coarse grid according to the parameters given in the
    // input file.
    dealii::GridGenerator::hyper_cube (triangulation, -5, 5);
    
    triangulation.refine_global (parameters.get_integer ("Global mesh refinement steps"));
  }


  /**
   * Setup system matrices and vectors.
   */
  template <int dim>
  void CatProblem<dim>::setup_system ()
  {
    dealii::TimerOutput::Scope time (timer, "setup system");

    // Determine locally relevant DoFs.
    dof_handler.distribute_dofs (fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    dealii::DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

    // Initialise values.
    solution_value.resize (n_pairs, 0.);
    
    // Initialise distributed vectors.
    locally_relevant_solution.resize (n_pairs);
    for (unsigned int i=0; i<n_pairs; ++i)
      locally_relevant_solution[i].reinit (locally_owned_dofs, locally_relevant_dofs,
					   mpi_communicator);

    // Setup hanging node constraints.
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    dealii::DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    dealii::DoFTools::make_zero_boundary_constraints (dof_handler, constraints);
    constraints.close ();

    // Finally, create a distributed sparsity pattern and initialise
    // the system- and mass-matrix from that.
    dealii::DynamicSparsityPattern dsp (locally_relevant_dofs);
    dealii::DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
    dealii::SparsityTools::distribute_sparsity_pattern (dsp,
							dof_handler.n_locally_owned_dofs_per_processor (),
							mpi_communicator,
							locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp,
			  mpi_communicator);
    
    mass_matrix.reinit (locally_owned_dofs, locally_owned_dofs,	dsp,
			mpi_communicator);
  }


  /**
   * Assemble system matrices and vectors.
   *
   * TODO Ideally, we would use a function like this:
   *
   * dealii::MatrixCreator::create_mass_matrix (dof_handler, quadrature_rule, system_matrix, 1, constraints);
   *
   * however no such thing currently exists in the deal.II library for
   * parallel matrices and vectors. Instead, the mass matrix and right
   * hand side vector are assembled by hand in functions defined in
   * the namepsaces, sarah::MatrixCreator and sarah::VectorCreator,
   * respectively.
   */
  template <int dim>
  void
  CatProblem<dim>::assemble_system ()
  {
    dealii::TimerOutput::Scope time (timer, "assemble system");

    // Define quadrature rule to be used.
    const dealii::QGauss<dim> quadrature_formula (3);

    sarah::MatrixCreator::create_mass_matrix<dim> (fe, dof_handler, quadrature_formula,
						   mass_matrix, constraints,
						   mpi_communicator);
    
    dealii::FEValues<dim> fe_values (fe, quadrature_formula,
				     dealii::update_values            |
				     dealii::update_gradients         |
				     dealii::update_quadrature_points |
				     dealii::update_JxW_values);
    
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    
    dealii::FullMatrix<double> cell_system_matrix (dofs_per_cell, dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    dealii::FunctionParser<dim> potential;
    potential.initialize (dealii::FunctionParser<dim>::default_variable_names (),
                          parameters.get ("Potential"),
                          typename dealii::FunctionParser<dim>::ConstMap ());
    
    std::vector<double> potential_values (n_q_points);
    
    typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active (),
      endc = dof_handler.end ();
    
    for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        cell_system_matrix = 0;
	
        potential.value_list (fe_values.get_quadrature_points (),
                              potential_values);
	
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                cell_system_matrix (i, j)
		  += (0.5                               *
		      fe_values.shape_grad (i, q_point) *
		      fe_values.shape_grad (j, q_point)
		      +
		      potential_values[q_point]          *
		      fe_values.shape_value (i, q_point) *
		      fe_values.shape_value (j, q_point)
		      )
		  * fe_values.JxW (q_point);
              }
	
        cell->get_dof_indices (local_dof_indices);
	
        constraints
	  .distribute_local_to_global (cell_system_matrix,
				       local_dof_indices,
				       system_matrix);
      }
    
    system_matrix.compress (dealii::VectorOperation::add);
    
  }
  

  /**
   * Solve the linear algebra system.
   */
  template <int dim>
  unsigned int
  CatProblem<dim>::solve ()
  {
    dealii::TimerOutput::Scope time (timer, "solve");
    
    std::vector<dealii::PETScWrappers::MPI::Vector> completely_distributed_solution;
    completely_distributed_solution.resize (n_pairs);
    for (unsigned int i=0; i<n_pairs; ++i)
      completely_distributed_solution[i].reinit (locally_owned_dofs,
						 mpi_communicator);
    
    // Solve using KrylovSchur
    dealii::SolverControl solver_control (dof_handler.n_dofs (), 1e-06);
    dealii::SLEPcWrappers::SolverKrylovSchur solver (solver_control, mpi_communicator);    

    solver.set_which_eigenpairs (EPS_SMALLEST_REAL);
    solver.set_problem_type (EPS_GHEP);
    
    solver.solve (system_matrix, mass_matrix, solution_value, completely_distributed_solution,
		  completely_distributed_solution.size ());

    // Ensure that all ghost elements are also copied as necessary.
    for (unsigned int i=0; i<n_pairs; ++i)
      constraints.distribute (completely_distributed_solution[i]);

    for (unsigned int i=0; i<n_pairs; ++i)
      locally_relevant_solution[i] = completely_distributed_solution[i];

    // Return the number of iterations (last step) of the solve.
    return solver_control.last_step ();
  }


  /**
   * Output results, ie., finite element functions and derived
   * quantitites for this cycle..
   */
  template <int dim>
  void
  CatProblem<dim>::output_results (const unsigned int cycle)
  {
    dealii::TimerOutput::Scope time (timer, "output_results");
    
    pcout << "   Values:"
	  << std::endl;
    for (unsigned int i=0; i<solution_value.size (); ++i)
      pcout << "      " << i << ": " << solution_value[i]
	    << std::endl;
    
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution[0], "wavefunction");
    
    dealii::Vector<double> projected_potential (dof_handler.n_dofs ());
    {
      dealii::FunctionParser<dim> potential;
      potential.initialize (dealii::FunctionParser<dim>::default_variable_names (),
                            parameters.get ("Potential"),
                            typename dealii::FunctionParser<dim>::ConstMap ());
      dealii::VectorTools::interpolate (dof_handler, potential, projected_potential);
    }
    data_out.add_data_vector (projected_potential, "interpolated_potential");
    
    dealii::Vector<float> subdomain (triangulation.n_active_cells ());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain (i) = triangulation.locally_owned_subdomain ();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches ();
    
    const std::string filename = ("wavefunction-" +
                                  dealii::Utilities::int_to_string (cycle, 2) +
                                  "." +
                                  dealii::Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain (), 4));

    std::ofstream output ((filename + ".vtu").c_str ());
    data_out.write_vtu (output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 1)
      {
	std::vector<std::string> filenames;
	
	for (unsigned int i=0;
	     i<dealii::Utilities::MPI::n_mpi_processes (mpi_communicator);
	     ++i)
	  filenames.push_back ("wavefunction-" +
			       dealii::Utilities::int_to_string (cycle, 2) +
			       "." +
			       dealii::Utilities::int_to_string (i, 4) +
			       ".vtu");
	std::ofstream master_output (("wavefunction-" +
				      dealii::Utilities::int_to_string (cycle, 2) +
				      ".pvtu").c_str ());

	data_out.write_pvtu_record (master_output, filenames);
      }

    {
      std::ostringstream filename;
      filename << "grid-" << cycle << ".gpl";
      std::ofstream output (filename.str ().c_str ());
      dealii::GridOut grid_out;
      grid_out.write_gnuplot (triangulation, output);
    }

  }


  /**
   * Refine grid based on Kelly's error estimator working on the
   * material id (solution vector).
   */
  template <int dim>
  void CatProblem<dim>::refine_grid ()
  {
    dealii::TimerOutput::Scope time (timer, "refine grid");

    dealii::Vector<float> estimated_error_per_cell (triangulation.n_active_cells ());
    
    dealii::KellyErrorEstimator<dim>::estimate (dof_handler, dealii::QGauss<dim-1>(4),
						typename dealii::FunctionMap<dim>::type (),
						locally_relevant_solution[0],
						estimated_error_per_cell);

    dealii::parallel::distributed::GridRefinement::
      refine_and_coarsen_fixed_number (triangulation,
				       estimated_error_per_cell,
				       0.250, 0.025);

    triangulation.execute_coarsening_and_refinement ();
  }
  
  
  /**
   * Run the application in the order specified.
   */
  template <int dim>
  void
  CatProblem<dim>::run ()
  {
    const unsigned int n_cycles = 5;
    
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        pcout << "CatProblem:: Cycle " << cycle << ':'
	      << std::endl;

	if (cycle==0)
	  make_coarse_grid ();

	else
	  // refine_grid ();
	  triangulation.refine_global ();
	
	pcout << "   Number of active cells:       "
	      << triangulation.n_global_active_cells ()
	      << std::endl;
	
	setup_system ();

	pcout << "   Number of degrees of freedom: "
	      << dof_handler.n_dofs ()
	      << std::endl;
	
	assemble_system ();

	const unsigned int n_iterations = solve ();

	pcout << "   Solved in " << n_iterations
	      << " iterations."
	      << std::endl;

	// Output results if the number of processes is less than or
	// equal to 32.
	if (dealii::Utilities::MPI::n_mpi_processes (mpi_communicator) <= 32)
	  output_results (cycle);

	// timer.print_summary ();
        pcout << std::endl;
	
      } // for cycle<n_cycles
  } 
  
} // namespace sarah


/**
 * Main function: Initialise problem and run it.
 */
int main (int argc, char *argv[])
{
  // Initialise MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv, 1);
  
  try
    {
      sarah::CatProblem<2> material_id ("step-0.prm");
      material_id.run ();
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
