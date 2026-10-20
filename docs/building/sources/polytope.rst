.. _sources-polytope:

##########
 polytope
##########

The ``polytope`` source retrieves full fields through the `Polytope`_ web
service, using the ``polytope`` source of ``earthkit-data``. It requires
the ``polytope-client`` package, and an account with access to the
collection being requested.

The ``request`` block follows the MARS language, with the exception of
the ``date`` and ``time`` keys, which are filled in from the dates of the
dataset. If ``step`` is not given, it defaults to 0.

.. literalinclude:: yaml/polytope.yaml
   :language: yaml

``collection``
   The Polytope collection (default: ``destination-earth``).

``address``
   The address of the Polytope server, if it is not the default one.

``flavour``
   Optional rules to rename or remap the fields, see
   ``anemoi.transform.flavour.RuleBasedFlavour``.

``grid_definition``
   Optional grid to attach to the fields, see
   ``anemoi.transform.grids.grid_registry``.

Any other key is passed on to ``earthkit.data.from_source("polytope",
...)``.

.. note::

   This source has not been tested against the DestinE Polytope service
   yet. Authentication, the handling of ``date``, ``time`` and ``step``
   for on-demand runs, and the request size limits still need to be
   checked.

.. _polytope: https://polytope.readthedocs.io/
