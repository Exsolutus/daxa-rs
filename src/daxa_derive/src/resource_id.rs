
use proc_macro::TokenStream;
use quote::quote;
use syn;

use bitfield::*;



pub fn impl_resource_id(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    
    validate_fields(&ast.data);

    // Implement ResourceId trait
    let gen = quote! {
        bitfield_bitrange! { struct #name(u32) }
        
        impl ResourceId for #name {
            bitfield_fields! {
                u32;
                #[inline]
                index, set_index: 23, 0;
                #[inline]
                u8, version, set_version: 31, 24;
            }

            #[inline]
            fn is_empty(&self) -> bool {
                self.version() == 0
            }
        }
    };
    gen.into()
}

fn validate_fields(data: &syn::Data) {
    match data {
        syn::Data::Struct(ref data) => {
            match data.fields {
                syn::Fields::Unnamed(ref fields) => {
                    //println!("{}, {:?}", fields.unnamed.len(), &fields.unnamed[0].ty);
                    if fields.unnamed.len() != 1 || !is_type(&fields.unnamed[0].ty, "u32") {
                        panic!("Structs deriving ResourceId should have a single unnamed u32 field.");
                    }
                },
                _ => unimplemented!("Structs deriving ResourceId should have no named fields.")
            }
        },
        _ => unimplemented!("Types deriving ResourceId should be structs.")
    };
}

fn is_type(ty: &syn::Type, target: &str) -> bool {
    if let syn::Type::Path(ref p) = ty {
        return p.path.segments.len() == 1 && p.path.segments[0].ident == target;
    } else {
        false
    }
}