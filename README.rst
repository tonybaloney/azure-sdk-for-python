Microsoft Azure SDK for Python
==============================

.. image:: http://www.quantifiedcode.com/api/v1/project/7e8a70bc3a1d445b95561f55712fee5a/badge.svg
        :target: http://www.quantifiedcode.com/app/project/7e8a70bc3a1d445b95561f55712fee5a

.. image:: https://travis-ci.org/Azure/azure-sdk-for-python.svg?branch=master
    :target: https://travis-ci.org/Azure/azure-sdk-for-python


This project provides a set of Python packages that make it easy to
access the Microsoft Azure components such as ServiceManagement, Storage\*, and ServiceBus.

The SDK supports Python 2.7, 3.3, 3.4 and 3.5.

\*Looking for the Azure Storage client library?  It moved to a `new GitHub repository <https://github.com/Azure/azure-storage-python>`__.

See important information if you're currently using this SDK < 1.0 in `this issue <https://github.com/Azure/azure-sdk-for-python/issues/440>`__.


DISCLAIMER
==========

This is an alpha release. Future releases may introduce some breaking changes.

Some of the new generated libraries have not yet been tested extensively, and some have known issues (such as azure-mgmt-web).

Our goal is to release a stable version by the end of March 2016.  Please send us your feedback!

Features
========

-  `Storage Runtime - Table, Blob, File, Queue </azure-storage-python>`__

-  `Service Bus Runtime </azure-servicebus>`__

-  `Service Management </azure-servicemanagement-legacy>`__

-  `Azure Active Directory Graph RBAC </azure-graphrbac>`__

-  `Resource Management </azure-mgmt>`__

   -  `Authorization </azure-mgmt-authorization>`__
   -  `CDN </azure-mgmt-cdn>`__
   -  `Compute </azure-mgmt-compute>`__
   -  Apps
   
       -  `Logic Apps </azure-mgmt-logic>`__
       -  `Web Apps </azure-mgmt-web>`__

   -  `Network </azure-mgmt-network>`__
   -  `Notification Hubs </azure-mgmt-notificationhubs>`__
   -  `Redis Cache </azure-mgmt-redis>`__
   -  `Resource </azure-mgmt-resource>`__
   -  `Scheduler </azure-mgmt-scheduler>`__
   -  `Storage </azure-mgmt-storage>`__


Installation
============

To install a bundle that includes all of the Azure client libraries listed above, see the `azure <https://github.com/Azure/azure-sdk-for-python/tree/master/azure>`__  bundle package.


Usage
=====

For detailed documentation, please view our `documentation on ReadTheDocs <http://azure-sdk-for-python.readthedocs.org>`__.

For further examples please visit the `Microsoft Azure Python Developer Center <http://azure.microsoft.com/en-us/develop/python/>`__.


Need Help?
==========

Be sure to check out the Microsoft Azure `Developer Forums on Stack Overflow <http://go.microsoft.com/fwlink/?LinkId=234489>`__
if you have trouble with the provided code.


Contribute Code or Provide Feedback
===================================

If you would like to become an active contributor to this project please
follow the instructions provided in `Microsoft Azure Projects Contribution Guidelines <http://azure.github.io/guidelines/>`__.

If you encounter any bugs with the library please file an issue in the
`Issues <https://github.com/Azure/azure-sdk-for-python/issues>`__
section of the project.


Learn More
==========

`Microsoft Azure Python Developer Center <http://azure.microsoft.com/en-us/develop/python/>`__
