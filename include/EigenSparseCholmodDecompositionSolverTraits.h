/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __EigenSparseCholmodDecompositionSolver_h
#define __EigenSparseCholmodDecompositionSolver_h

#include <vector>
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/CholmodSupport"

template< typename T = double >
class EigenSparseCholmodDecompositionSolverTraits
{
public:
  typedef T                                             ValueType;
  typedef Eigen::SparseMatrix< ValueType >              MatrixType;
  typedef Eigen::Matrix< ValueType, Eigen::Dynamic, 1 > VectorType;
  typedef Eigen::Triplet< ValueType >                   TripletType;
  typedef std::vector< TripletType >                    TripletVectorType;
  typedef Eigen::CholmodDecomposition< MatrixType > SolverType;

  /** \return true (it is a direct solver) */
  static bool IsDirectSolver()
  {
    return true;
  }

  /** \brief initialize a square sparse matrix of size iN x iN */
  static MatrixType InitializeSparseMatrix(const unsigned int & iN)
  {
    return MatrixType( iN, iN );
  }

  /** \brief initialize a sparse matrix of size iRow x iCol */
  static MatrixType InitializeSparseMatrix(const unsigned int & iRow, const unsigned int& iCol)
  {
    return MatrixType( iRow, iCol );
  }

  /** \brief initialize a vector of size iN */
  static VectorType InitializeVector(const unsigned int & iN)
  {
    return VectorType(iN);
  }

  /** \brief iA[iR][iC] = iV */
  static void FillMatrix(MatrixType & iA, const unsigned int & iR, const unsigned int & iC, const ValueType & iV)
  {
    iA.insert( iR, iC ) = iV;
  }

  /** \brief iA[iR][iC] = iV, for each triplet (iR,iC,iV) */
  static void FillMatrixWithTriplets( MatrixType & iA, const TripletVectorType & triplets )
  {
    iA.setFromTriplets( triplets.begin(), triplets.end() );
  }

  /** \brief iA[iR][iC] += iV */
  static void AddToMatrix(MatrixType & iA, const unsigned int & iR, const unsigned int & iC, const ValueType & iV)
  {
    iA.coeffRef( iR, iC ) += iV;
  }

  /** \brief Solve the linear system \f$ iA \cdot oX = iB \f$ */
  static bool Solve(const MatrixType & iA, const VectorType & iB, VectorType & oX)
  {
    SolverType solver;
    solver.analyzePattern( iA );
    solver.factorize( iA );
    oX = solver.solve( iB );

    return solver.info() == Eigen::Success;;
  }

  /** \brief Solve the linear systems: \f$ iA \cdot oX = iBx \f$, \f$ iA \cdot oY = iBy \f$, \f$ iA \cdot oZ = iBz \f$ */
  static bool Solve(const MatrixType & iA,
             const VectorType & iBx, const VectorType & iBy, const VectorType & iBz,
             VectorType & oX, VectorType & oY, VectorType & oZ )
  {
    SolverType solver;
    solver.analyzePattern( iA );
    solver.factorize( iA );
    oX = solver.solve( iBx );
    oY = solver.solve( iBy );
    oZ = solver.solve( iBz );

    return solver.info() == Eigen::Success;;
  }

  /** \brief Solve the linear systems: \f$ iA \cdot oX = iBx \f$, \f$ iA \cdot oY = iBy \f$ */
  static bool Solve(const MatrixType & iA,
             const VectorType & iBx, const VectorType & iBy,
             VectorType & oX, VectorType & oY)
  {
    SolverType solver;
    solver.analyzePattern( iA );
    solver.factorize( iA );
    oX = solver.solve( iBx );
    oY = solver.solve( iBy );

    return solver.info() == Eigen::Success;;
  }

  /** \brief Solve the linear system \f$ iA \cdot oX = iB \f$ with the already factored iA matrix */
  static void Solve( const SolverType & solver, const VectorType & iB, VectorType & oX)
  {
    oX = solver.solve( iB );
  }

  /** \brief Solve the linear systems: \f$ iA \cdot oX = iBx \f$, \f$ iA \cdot oY = iBy \f$, \f$ iA \cdot oZ = iBz \f$ with the already factored iA matrix */
  static void Solve( const SolverType & solver,
             const VectorType & iBx, const VectorType & iBy, const VectorType & iBz,
             VectorType & oX, VectorType & oY, VectorType & oZ )
  {
    oX = solver.solve( iBx );
    oY = solver.solve( iBy );
    oZ = solver.solve( iBz );
  }

  /** \brief Solve the linear systems: \f$ iA \cdot oX = iBx \f$, \f$ iA \cdot oY = iBy \f$ with the already factored iA matrix */
  static void Solve( const SolverType & solver,
             const VectorType & iBx, const VectorType & iBy,
             VectorType & oX, VectorType & oY)
  {
    oX = solver.solve( iBx );
    oY = solver.solve( iBy );
  }
};

#endif
