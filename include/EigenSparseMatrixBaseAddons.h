#ifndef __EigenSparseMatrixBaseAddons_h
#define __EigenSparseMatrixBaseAddons_h

template< typename OtherDerived >
inline void
mult( const MatrixBase< OtherDerived > & iM,
      MatrixBase< OtherDerived > & oM )
{
  //template<typename OtherDerived>
  //const typename SparseDenseProductReturnType<Derived,OtherDerived>::Type
  //operator*(const MatrixBase<OtherDerived> &other) const;
  oM = *this * iM;
}

#endif
