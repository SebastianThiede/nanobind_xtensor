#ifndef NANOBIND_XTENSOR_CASTER
#define NANOBIND_XTENSOR_CASTER

#include "xtensor/xtensor.hpp"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include <type_traits>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

namespace nb = nanobind;

// detectors -------------------------------------------------------------------

template<typename T>
struct is_xtensor_fixed : std::false_type {};

template<class ET, class S, xt::layout_type L, bool SH, class Tag>
struct is_xtensor_fixed<xt::xfixed_container<ET, S, L, SH, Tag>> : std::true_type {};

template<typename T>
constexpr bool is_xtensor_fixed_v = is_xtensor_fixed<T>::value;



template<typename T>
struct is_xtensor : std::false_type {};

template<class EC, std::size_t N, xt::layout_type L, class Tag>
struct is_xtensor<xt::xtensor_container<EC, N, L, Tag>> : std::true_type {};

template<typename T>
constexpr bool is_xtensor_v = is_xtensor<T>::value;



template<typename T>
struct is_xarray : std::false_type {};

template<class EC, xt::layout_type L, class SC, class Tag>
struct is_xarray<xt::xarray_container<EC, L, SC, Tag>> : std::true_type {};

template<typename T>
constexpr bool is_xarray_v = is_xarray<T>::value;

// utilities -------------------------------------------------------------------

template<typename T>
struct nb_shape_param {
    using value = nb::detail::unused;
};

template<class ET, xt::layout_type L, bool SH, class Tag, std::size_t... Sizes>
struct nb_shape_param<xt::xfixed_container<ET, xt::xshape<Sizes...>, L, SH, Tag>> {
    using value = nb::shape<Sizes...>;
};

template<class EC, std::size_t N, xt::layout_type L, class Tag>
struct nb_shape_param<xt::xtensor_container<EC, N, L, Tag>> {
    using value = nb::ndim<N>;
};

template<typename T>
using nb_shape_param_v = typename nb_shape_param<T>::value;



// Get scalar type of an xexpression
template<typename T>
using xexpression_scalar_t = typename std::decay_t<T>::value_type;

// Get the nanobind array type for an xtensor
template<typename T, typename... additional>
struct NDArray_for_xtensor {
    using scalar_t = xexpression_scalar_t<T>;
    using type = ndarray<
        scalar_t,
        std::conditional_t<T::static_layout == xt::layout_type::row_major,
            nb::c_contig,
            std::conditional_t<T::static_layout == xt::layout_type::column_major,
                nb::f_contig, 
                nb::any_contig
            >
        >,
        nb_shape_param_v<T>,
        additional...
    >;
};

template<typename T, typename... additional>
using NDArray_for_xtensor_t = typename NDArray_for_xtensor<T, additional...>::type;

// const version
template<typename T, typename... additional>
struct constNDArray_for_xtensor {
    using scalar_t = std::add_const_t<xexpression_scalar_t<T>>;
    using type = ndarray<
        scalar_t,
        std::conditional_t<T::static_layout == xt::layout_type::row_major,
            nb::c_contig,
            std::conditional_t<T::static_layout == xt::layout_type::column_major,
                nb::f_contig, 
                nb::any_contig
            >
        >,
        nb_shape_param_v<T>,
        additional...
    >;
};
template<typename T, typename... additional>
using constNDArray_for_xtensor_t = typename constNDArray_for_xtensor<T, additional...>::type;

// Bind xtensor ----------------------------------------------------------------
template<typename T>
struct type_caster<T, enable_if_t<
                        (is_xtensor_fixed_v<T> || is_xtensor_v<T> || is_xarray_v<T>) &&
                        is_ndarray_scalar_v<xexpression_scalar_t<T>>
                      >>
{
    using self_t = T;
    using self_shape_t = std::conditional_t<is_xtensor_fixed_v<T>, std::array<std::size_t, T::rank>, typename T::shape_type>;
    using NDArray_t = NDArray_for_xtensor_t<T>;
    using constNDArray_t = constNDArray_for_xtensor_t<T>;
    using NDArray_return_t = NDArray_for_xtensor_t<T, nb::numpy>;

    NB_TYPE_CASTER(self_t, make_caster<NDArray_t>::Name)

    bool from_python(
        handle src,
        uint8_t flags,
        cleanup_list* cleanup
    ) noexcept
    {
        make_caster<constNDArray_t> caster;

        flags = flags_for_local_caster<constNDArray_t>(flags);

        bool succ = caster.from_python(src, flags, cleanup);
        if (!succ)
            return false;
        
        constNDArray_t& NDArray = caster.value;

        value = self_t();

        if constexpr (!is_xtensor_fixed_v<self_t>)
        {
            self_shape_t shape{NDArray.ndim()};
            for (std::size_t i = 0; i < NDArray.ndim(); i++)
            {
                shape[i] = NDArray.shape(i);
            }
            value.resize(shape);
        };

        memcpy(value.data(), NDArray.data(), NDArray.size() * sizeof(typename constNDArray_for_xtensor<self_t>::scalar_t));

        return true;
    };

    static handle from_cpp(
        const self_t& tensor,
        rv_policy policy,
        cleanup_list *cleanup
    ) noexcept
    {
        void* ptr = (void *) tensor.data();

        switch (policy) {
            case rv_policy::automatic:
                policy = rv_policy::move;
                break;

            case rv_policy::automatic_reference:
                policy = rv_policy::reference;
                break;

            default: // leave policy unchanged
                break;
        }

        object owner = steal(handle());
        if (policy == rv_policy::move) {
            self_t *temp = new self_t(std::move(tensor));
            owner = capsule(temp, [](void *p) noexcept { delete (self_t *) p; });
            ptr = temp->data();
            policy = rv_policy::reference;
        } else if (policy == rv_policy::reference_internal && cleanup->self()) {
            owner = borrow(cleanup->self());
            policy = rv_policy::reference;
        }

        object o = steal(
            make_caster<NDArray_return_t>::from_cpp(
                NDArray_return_t(
                    ptr,
                    tensor.dimension(),
                    tensor.shape().begin(),
                    owner,
                    (tensor.dimension() == 0) ? nullptr : tensor.strides().begin()
                    ),
                policy,
                cleanup
            )
        );

        return o.release();
    };
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif
