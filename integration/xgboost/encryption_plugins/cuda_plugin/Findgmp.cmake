####################################################
# Looking for main header file in standard locations
find_path(gmpc_INCLUDE_DIR gmp.h)
find_path(gmpxx_INCLUDE_DIR gmpxx.h)

############################################
# Looking for binaries in standard locations
find_library(gmpc_LIBRARY NAMES gmp libgmp)
find_library(gmpxx_LIBRARY NAMES gmpxx libgmpxx)

######################################################################################
# QUESTION: IS ALL THAT NECESSARY OR find_package_handle_standard_args DOES THE JOB? #
######################################################################################
IF (gmpc_INCLUDE_DIR STREQUAL "gmpc_INCLUDE_DIR-NOTFOUND")
	MESSAGE(WARNING "GMP c headers not found")
	SET(GMP_DETECTION_ERROR TRUE)
ELSEIF(gmpxx_INCLUDE_DIR STREQUAL "gmpxx_INCLUDE_DIR-NOTFOUND")
	MESSAGE(WARNING "GMP c++ headers not found")
	SET(GMP_DETECTION_ERROR TRUE)
ELSEIF(gmpc_LIBRARY STREQUAL "gmpc_LIBRARY-NOTFOUND")
	MESSAGE(WARNING "GMP c library not found")
	SET(GMP_DETECTION_ERROR TRUE)
ELSEIF(gmpxx_LIBRARY STREQUAL "gmpxx_LIBRARY-NOTFOUND")
	MESSAGE(WARNING "GMP c++ library not found")
	SET(GMP_DETECTION_ERROR TRUE)
ENDIF()

IF (NOT GMP_DETECTION_ERROR)

	mark_as_advanced(gmpc_INCLUDE_DIR gmpc_LIBRARY gmpxx_INCLUDE_DIR gmpxx_LIBRARY)

	#############################
	# Setting find_package output
	#	gmp_FOUND
	# Cache variables
	#	gmp_INCLUDE_DIR
	#	gmp_LIBRARY
	# CMakeLists variables
	#	gmp_INCLUDE_DIRS
	#	gmp_LIBRARIES
	include( FindPackageHandleStandardArgs )
	find_package_handle_standard_args(gmp REQUIRED_VARS
		gmpc_LIBRARY
		gmpxx_LIBRARY
		gmpc_INCLUDE_DIR
		gmpxx_INCLUDE_DIR
		)
		
	IF (gmp_FOUND)

		SET(gmpc_LIBRARIES ${gmpc_LIBRARY})
		SET(gmpc_INCLUDE_DIRS ${gmpc_INCLUDE_DIR})
		SET(gmpxx_LIBRARIES ${gmpxx_LIBRARY})
		SET(gmpxx_INCLUDE_DIRS ${gmpxx_INCLUDE_DIR})

		##################################
		# Setting gmp::gmp
		IF (NOT TARGET gmp::gmpc)
			add_library(gmp::gmpc UNKNOWN IMPORTED)
			set_target_properties(gmp::gmpc PROPERTIES
				IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
				IMPORTED_LOCATION "${gmpc_LIBRARY}"
				INTERFACE_INCLUDE_DIRECTORIES "${gmpc_INCLUDE_DIR}"
				)
		ENDIF()
		#SET(GMPC_TARGET "gmp::gmpc")
		IF (NOT TARGET gmp::gmpxx)
			add_library(gmp::gmpxx UNKNOWN IMPORTED)
			set_target_properties(gmp::gmpxx PROPERTIES
				IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
				IMPORTED_LOCATION "${gmpxx_LIBRARY}"
				INTERFACE_INCLUDE_DIRECTORIES "${gmpxx_INCLUDE_DIR}"
				)
		ENDIF()
		#SET(GMPXX_TARGET "gmp::gmpxx")
		IF (NOT TARGET gmp::gmp)
			add_library(gmp::gmp INTERFACE IMPORTED)
			#SET(GMP_TARGET "${GMPC_TARGET};${GMPXX_TARGET}")
			set_target_properties(gmp::gmp PROPERTIES
				LINK_INTERFACE_LIBRARIES "gmp::gmpc;gmp::gmpxx"
				IMPORTED_LOCATION "${gmpc_LIBRARY};${gmpxx_LIBRARY}")
		ENDIF()

	ENDIF()
	
ENDIF()

