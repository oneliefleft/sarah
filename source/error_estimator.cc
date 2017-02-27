// -----------------------------------------------------------------------------
// 
// BSD 2-Clause License
// 
// Copyright (c) 2017, sarah authors
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// -----------------------------------------------------------------------------

#include <sarah/error_estimator.h>

namespace sarah
{

  namespace ErrorEstimator
  {
    
    template<int dim, int spacedim = dim, typename ValueType = double>
    void
    estimate (const dealii::FiniteElement<dim,spacedim> &finite_element,
	      const dealii::DoFHandler<dim,spacedim>    &dof_handler,
	      const dealii::Quadrature<dim>             &quadrature,
	      const dealii::PETScWrappers::MPI::Vector  &fe_function,
	      dealii::Vector<ValueType>                 &error_per_cell,
	      MPI_Comm                                  &mpi_communicator)
    {
      dealii::FEValues<dim> fe_values (finite_element, quadrature,
				       dealii::update_values            |
				       dealii::update_quadrature_points |
				       dealii::update_JxW_values);
      
      const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
      const unsigned int n_q_points    = quadrature.size ();
      
      std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<ValueType> function_values (n_q_points);
      
      // Let there be a cell identification number. This
      // identification has nothing to do with the internal
      // description of the cell, though it probably should.
      //
      // Perhaps there is a dealii::type::??? for this?
      unsigned int cell_id = 0;
      
      typename dealii::DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active (),
	endc = dof_handler.end ();
      
      for (; cell!=endc; ++cell, ++cell_id)
	if (cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_communicator))
	  {
	    fe_values.reinit (cell);

	    fe_values.get_function_values (fe_function, function_values);
	    
	    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    // Integrate volume of the function in this cell.
		    error_per_cell (cell_id) +=
		      function_values[q_point]          *
		      fe_values.shape_value (j,q_point) *
		      fe_values.JxW (q_point);
		  }

	    // This should *never* happen, but sometimes it does....
	    error_per_cell (cell_id) = std::fabs (error_per_cell (cell_id));

	  } // cell!=endc
    }

    template<int dim, int spacedim = dim, typename ValueType = double>
    void
    estimate (const dealii::FiniteElement<dim,spacedim> &finite_element,
	      const dealii::DoFHandler<dim,spacedim>    &dof_handler,
	      const dealii::Quadrature<dim>             &quadrature,
	      const dealii::FunctionParser<dim>         &function_parser,
	      dealii::Vector<ValueType>                 &error_per_cell,
	      MPI_Comm                                  &mpi_communicator)
    {
      dealii::FEValues<dim> fe_values (finite_element, quadrature,
				       dealii::update_values            |
				       dealii::update_quadrature_points |
				       dealii::update_JxW_values);
      
      const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
      const unsigned int n_q_points    = quadrature.size ();
      
      std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<ValueType> function_values (n_q_points);
      
      // Let there be a cell identification number. This
      // identification has nothing to do with the internal
      // description of the cell, though it probably should.
      //
      // Perhaps there is a dealii::type::??? for this?
      unsigned int cell_id = 0;
      
      typename dealii::DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active (),
	endc = dof_handler.end ();
      
      for (; cell!=endc; ++cell, ++cell_id)
	if (cell->subdomain_id () == dealii::Utilities::MPI::this_mpi_process (mpi_communicator))
	  {
	    fe_values.reinit (cell);

	    function_parser.value_list (fe_values.get_quadrature_points (),
					function_values);
	    
	    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		  {
		    // Integrate volume of the function in this cell.
		    error_per_cell (cell_id) +=
		      function_values[q_point]          *
		      fe_values.shape_value (j,q_point) *
		      fe_values.JxW (q_point);
		  }

	    // This should *never* happen, but sometimes it does....
	    error_per_cell (cell_id) = std::fabs (error_per_cell (cell_id));

	  } // cell!=endc
    }
    
  } // namepsace ErrorEstimator
  
} // namepsace sarah

template void
sarah::ErrorEstimator::estimate<3, 3, double> (const dealii::FiniteElement<3,3>         &,
					       const dealii::DoFHandler<3,3>            &,
					       const dealii::Quadrature<3>              &,
					       const dealii::PETScWrappers::MPI::Vector &,
					       dealii::Vector<double>                   &,
					       MPI_Comm                                 &);

template void
sarah::ErrorEstimator::estimate<2, 2, double> (const dealii::FiniteElement<2,2>         &,
					       const dealii::DoFHandler<2,2>            &,
					       const dealii::Quadrature<2>              &,
					       const dealii::PETScWrappers::MPI::Vector &,
					       dealii::Vector<double>                   &,
					       MPI_Comm                                 &);

template
void
sarah::ErrorEstimator::estimate<2, 2, double> (const dealii::FiniteElement<2,2> &,
					       const dealii::DoFHandler<2,2>    &,
					       const dealii::Quadrature<2>      &,
					       const dealii::FunctionParser<2>  &,
					       dealii::Vector<double>           &,
					       MPI_Comm                         &);
